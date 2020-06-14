import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import pickle

import numpy as np
from ..config import STYLEGAN_WEIGHTS_PATH

"""
Credit to
https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
where this file is almost entirely copied from.
"""


class MyLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""
    def __init__(self, input_size, output_size, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size**(-0.5) # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class MyConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""
    def __init__(self, input_channels, output_channels, kernel_size, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True,
                intermediate=None, upscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5) # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x, patch_coords=None, patch_dim=None):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        # condition `x.shape[1] < 512` equivalent to `min(x.shape[2:]) * 2 >= 128` when full image is being generated
        if self.upscale is not None and x.shape[1] < 512:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, (1,1,1,1))
            w = w[:, :, 1:, 1:]+ w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1)-1)//2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        # select glimpses of x
        if patch_coords is not None:
            B, N, H, W = x.shape
            x_glimpses = []
            for i, (px, py) in enumerate(patch_coords):

                glimpse = torch.zeros(1, N, patch_dim, patch_dim)
                patch_minx = max(0, -px)
                patch_maxx = min(patch_dim, H-px)
                patch_miny = max(0, -py)
                patch_maxy = min(patch_dim, W-py)
                img_minx = max(0, px)
                img_maxx = min(px+patch_dim, H)
                img_miny = max(0, py)
                img_maxy = min(py+patch_dim, W)

                layer = 0 if B == 1 else i
                patch = torch.zeros(N, patch_dim, patch_dim).to(x.device)
                patch[:, patch_minx:patch_maxx, patch_miny:patch_maxy] = x[layer, :, img_minx:img_maxx, img_miny:img_maxy]
                x_glimpses.append(patch)

            x = torch.stack(x_glimpses, dim=0)

        if not have_convolution and self.intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size//2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size//2)

        if self.intermediate is not None:
            x = self.intermediate(x)
        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noise layers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = MyLinear(latent_size,
                            channels * 2,
                            gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.lin(latent) # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class BlurLayer(nn.Module):
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        kernel=[1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2)-1)/2),
            groups=x.size(1)
        )
        return x


def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
    return x


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor
    def forward(self, x):
        return upscale2d(x, factor=self.factor, gain=self.gain)


class G_mapping(nn.Sequential):
    def __init__(self, nonlinearity='lrelu', use_wscale=True):
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        layers = [
            ('pixel_norm', PixelNormLayer()),
            ('dense0', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense0_act', act),
            ('dense1', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense1_act', act),
            ('dense2', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense2_act', act),
            ('dense3', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense3_act', act),
            ('dense4', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense4_act', act),
            ('dense5', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense5_act', act),
            ('dense6', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense6_act', act),
            ('dense7', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense7_act', act)
        ]
        super().__init__(OrderedDict(layers))

    def forward(self, x):
        x = super().forward(x)
        # Broadcast
        x = x.unsqueeze(1).expand(-1, 18, -1)
        return x


class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.register_buffer('avg_latent', avg_latent)
    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1)
        return torch.where(do_trunc, interp, x)


class FixedNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def set_mean_var(self, m, v):
        self.m = m.view(1, self.channels, 1, 1)
        self.std = v.view(1, self.channels, 1, 1)**0.5

    def forward(self, x):
        x = (x-self.m)/(self.std + 1e-5)
#        new_mean = x.mean(-1).mean(-1).mean(0)
#        new_var = ((x - new_mean.view(1, -1, 1, 1))**2).mean(-1).mean(-1).mean(0)
#        print("mean:", new_mean)
#        print("var:", new_var)
        return x


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""
    def __init__(self, channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer, use_fixed_norm=False):
        super().__init__()
        layers = []
        if use_noise:
            layers.append(('noise', NoiseLayer(channels)))
        layers.append(('activation', activation_layer))
        if use_pixel_norm:
            layers.append(('pixel_norm', PixelNorm()))
        if use_instance_norm:
            Norm = FixedNorm2d if use_fixed_norm else nn.InstanceNorm2d
            layers.append(('instance_norm', Norm(channels)))
        self.top_epi = nn.Sequential(OrderedDict(layers))
        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class InputBlock(nn.Module):
    def __init__(self, nf, dlatent_size, const_input_layer, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            # called 'const' in tf
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = MyLinear(dlatent_size, nf*16, gain=gain/4, use_wscale=use_wscale) # tweak gain to match the official implementation of Progressing GAN
        self.epi1 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        self.conv = MyConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer)

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class GSynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer, n_convs, use_fixed_norm):
        # 2**res x 2**res # res = 3..resolution_log2
        super().__init__()
        self.n_convs = n_convs
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.conv0_up = MyConv2d(in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale,
                                 intermediate=blur, upscale=True)
        self.epi1 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer, use_fixed_norm=use_fixed_norm)
        self.conv1 = MyConv2d(out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer, use_fixed_norm=use_fixed_norm)

    def forward(self, x, dlatents_in_range, patch_coords=None, patch_dim=None):
        x = self.conv0_up(x, patch_coords=patch_coords, patch_dim=patch_dim)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        if patch_dim is not None:
            # strip away edges of patch (i.e. undo padding)
            _, _, h, w = x.shape
            m = self.n_convs
            x = x[:, :, m:h-m, m:w-m]
        return x


class G_synthesis(nn.Module):
    def __init__(self,
        dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
        num_channels        = 3,            # Number of output color channels.
        resolution          = 1024,         # Output resolution.
        fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
        fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
        fmap_max            = 512,          # Maximum number of feature maps in any layer.
        use_styles          = True,         # Enable style inputs?
        const_input_layer   = True,         # First layer is a learned constant?
        use_noise           = True,         # Enable noise inputs?
        randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu'
        use_wscale          = True,         # Enable equalized learning rate?
        use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
        use_instance_norm   = True,         # Enable instance normalization?
        dtype               = torch.float32,  # Data type to use for activations and outputs.
        blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
        split_res           = np.inf,       # Resolution at which split, if G_patch_synthesis inherited
        n_convs             = 0             # Number of convs to account for if splitting image. Makes no difference if not.
        ):

        super().__init__()
        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        num_layers = resolution_log2 * 2 - 2
        num_styles = num_layers if use_styles else 1
        torgbs = []
        blocks = []
        for res in range(2, resolution_log2 + 1):
            channels = nf(res-1)
            name = '{s}x{s}'.format(s=2**res)
            if res == 2:
                blocks.append((name,
                               InputBlock(channels, dlatent_size, const_input_layer, gain, use_wscale,
                                      use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))
            else:
                blocks.append((name,
                               GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, act, n_convs=n_convs, use_fixed_norm=res>=split_res)))
            last_channels = channels
        self.torgb = MyConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale)
        self.blocks = nn.ModuleDict(OrderedDict(blocks))

    def forward(self, dlatents_in):
        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        # lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)
        for i, m in enumerate(self.blocks.values()):
            if i == 0:
                x = m(dlatents_in[:, 2*i:2*i+2])
            else:
                x = m(x, dlatents_in[:, 2*i:2*i+2])
        rgb = self.torgb(x)
        return rgb


class G_patch_synthesis(G_synthesis):
    split_res = 10  # 10 is optimal - lower gives poor normalisations
    n_convs = 0     # I think 3 is right but higher should be fine

    def lower_dims(self, dim, log_res=10):
        if log_res == self.split_res:
            return [dim]
        dim_before_upsampling = int(dim/2)+1
        dim_below = dim_before_upsampling + 2*self.n_convs
        lower_levels = self.lower_dims(dim_below, log_res-1)
        lower_levels.append(dim)
        return lower_levels

    def lower_coords(self, x, y, log_res=10):
        """
        x, y:     global coordinates we require to be exactly right
        returns:  local coordinates for all lower layers to select these
               - can only be negative for first patch - we zero-pad if it is
        """
        if log_res == self.split_res:
            return [(x, y)]
        x_below = x//2 - self.n_convs  # //2 to undo upsample and then -2 to undo convolutions
        y_below = y//2 - self.n_convs
        all_local = self.lower_coords(x_below, y_below, log_res-1)
        global_x_after_upsampling = (x_below+self.n_convs)*2  # coordinates of `exact patch` relative to image origin
        global_y_after_upsampling = (y_below+self.n_convs)*2
        local_x = x - global_x_after_upsampling   # coordinates of lower layer relative to convolved/upsampled previous patch
        local_y = y - global_y_after_upsampling
        all_local.append((local_x, local_y))
        return all_local

    def __init__(self, *args, patch_dim=16, rescale_dim=224, **kwargs):
        super().__init__(*args, split_res=self.split_res, n_convs=self.n_convs, **kwargs)
        self.rescale_dim = rescale_dim
        self.top_dim = int(patch_dim*1024/rescale_dim+2)  # dimension at highest level
        self.dims = self.lower_dims(self.top_dim+2*self.n_convs, log_res=10)  # +4 to account for last convolutions

    def _iter_instance_norms(self):
        for block in self.blocks.values():
            yield block.epi1.top_epi.instance_norm
            yield block.epi2.top_epi.instance_norm

    def set_instance_norm_stats(self):
        means = pickle.load(open("stylegan-weights/instance-norm-means.p", 'rb'))
        vars = pickle.load(open("stylegan-weights/instance-norm-vars.p", 'rb'))
        for m, v, norm in zip(means, vars, self._iter_instance_norms()):
            if type(norm) == FixedNorm2d:
                norm.set_mean_var(m, v)

    def set_coords(self, coords):
        # calculate coordinates of patches to take
        self.top_coords = [(int(x*1024/self.rescale_dim)-self.n_convs, int(y*1024/self.rescale_dim)-self.n_convs)
                           for x, y in coords]
        coords = np.array([self.lower_coords(*top_coord)
                           for top_coord in self.top_coords])
        self.coords = coords.transpose(1, 0, 2)  # shape: (n_layers, n_patches, 2)

    def forward(self, dlatents_in):
        """ returns a 1024x1024 image such that, if scaled to rescale_dim and
            path_dim x patch_dim patch is observed, it looks right.
        """
        # only allow batch size of 1 for now
        assert dlatents_in.size(0) == 1

        # now start running neural net
        for i, m in enumerate(self.blocks.values()):
            res = i+2
            if i == 0:
                x = m(dlatents_in[:, 2*i:2*i+2])
            elif res < self.split_res:
                x = m(x, dlatents_in[:, 2*i:2*i+2])
            else:
                # take chunks out of x as specified
                # pretend we are batching it
                # res is what m will increase the resolution to
                x = m(x, dlatents_in[:, 2*i:2*i+2],
                      patch_coords=self.coords[res-self.split_res],
                      patch_dim=self.dims[res-self.split_res])
        rgb = self.torgb(x)

        # combine patches onto an image
        image = torch.zeros(1, 3, 1024, 1024).to(rgb.device)
        pdim = self.top_dim
        for (px, py), patch in zip(self.top_coords, rgb):
            image[0, :, px:px+pdim, py:py+pdim] = patch

        return image


def get_generator(cuda, patchy=False):
    """
    Returns generator in eval mode.
    """
    g_all = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        ('g_synthesis', G_patch_synthesis() if patchy else G_synthesis())
    ]))
    g_all.load_state_dict(torch.load(STYLEGAN_WEIGHTS_PATH))
    if patchy:
        # useful to have this here for testing
        g_all.g_synthesis.set_coords([(95, 80), (90, 110), (100, 126)])
        g_all.g_synthesis.set_instance_norm_stats()
    g_all.eval()
    device = 'cuda:0' if cuda else 'cpu'
    g_all.to(device)
    return g_all
