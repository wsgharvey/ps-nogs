"""
mostly copied from https://github.com/kkanshul/finegan
(FineGAN: Unsupervised Hierarchical Disentanglement for
 Fine-Grained Object Generation and Discovery)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mea.config import FINEGAN_WEIGHTS_PATH

# parameters from the FineGAN paper
GAN_Z_DIM = 100
GAN_R_NUM = 2
FINE_GRAINED_CATEGORIES = 200
SUPER_CATEGORIES = 20
GAN_GF_DIM = 64


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
padding=1, bias=False)


def convlxl(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=13, stride=1,
padding=1, bias=False)


def child_to_parent(child_c_code, classes_child, classes_parent):
    ratio = classes_child // classes_parent
    arg_parent = torch.argmax(child_c_code,  dim = 1).int() // ratio
    parent_c_code = torch.zeros([child_c_code.size(0), classes_parent]).cuda()
    for i in range(child_c_code.size(0)):
        parent_c_code[i][arg_parent[i]] = 1
    return parent_c_code


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

def sameBlock(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, c_flag):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.c_flag= c_flag

        if self.c_flag==1 :
            self.in_dim = GAN_Z_DIM + SUPER_CATEGORIES
        elif self.c_flag==2:
            self.in_dim = GAN_Z_DIM + FINE_GRAINED_CATEGORIES

        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU(),
        )
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        self.upsample5 = upBlock(ngf // 16, ngf // 16)


    def forward(self, z_code, code):
        in_code = torch.cat((code, z_code), 1)
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        out_code = self.upsample5(out_code)
        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, use_hrc = 1, num_residual=GAN_R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if use_hrc == 1: # For parent stage
            self.ef_dim = SUPER_CATEGORIES
        else:            # For child stage
            self.ef_dim = FINE_GRAINED_CATEGORIES
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim
        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.samesample = sameBlock(ngf, ngf // 2)

    def forward(self, h_code, code):
        s_size = h_code.size(2)
        code = code.view(-1, self.ef_dim, 1, 1)
        code = code.repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((code, h_code), 1)
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.samesample(out_code)
        return out_code


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class GET_MASK_G(nn.Module):
    def __init__(self, ngf):
        super(GET_MASK_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 1),
            nn.Sigmoid()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = GAN_GF_DIM
        self.define_module()
        self.scale_fimg = nn.UpsamplingBilinear2d(size = [126, 126])

    def define_module(self):

        #Background stage
        self.h_net1_bg = INIT_STAGE_G(self.gf_dim * 16, 2)
        self.img_net1_bg = GET_IMAGE_G(self.gf_dim) # Background generation network

        # Parent stage networks
        self.h_net1 = INIT_STAGE_G(self.gf_dim * 16, 1)
        self.h_net2 = NEXT_STAGE_G(self.gf_dim, use_hrc = 1)
        self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)  # Parent foreground generation network
        self.img_net2_mask= GET_MASK_G(self.gf_dim // 2) # Parent mask generation network

        # Child stage networks
        self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2, use_hrc = 0)
        self.img_net3 = GET_IMAGE_G(self.gf_dim // 4) # Child foreground generation network
        self.img_net3_mask = GET_MASK_G(self.gf_dim // 4) # Child mask generation network

    def forward(self, z_code, c_code,):

        p_code = child_to_parent(c_code, FINE_GRAINED_CATEGORIES, SUPER_CATEGORIES) # Obtaining the parent code from child code
        bg_code = c_code

        #Background stage
        h_code1_bg = self.h_net1_bg(z_code, bg_code)
        fake_img1 = self.img_net1_bg(h_code1_bg) # Background image
        # fake_img1_126 = self.scale_fimg(fake_img1) # Ppaer did this, we do not want to.
        bg_image = fake_img1

        #Parent stage
        h_code1 = self.h_net1(z_code, p_code)
        h_code2 = self.h_net2(h_code1, p_code)
        fake_img2_foreground = self.img_net2(h_code2) # Parent foreground
        fake_img2_mask = self.img_net2_mask(h_code2) # Parent mask
        ones_mask_p = torch.ones_like(fake_img2_mask)
        opp_mask_p = ones_mask_p - fake_img2_mask
        fg_masked2 = torch.mul(fake_img2_foreground, fake_img2_mask)
        bg_masked2 = torch.mul(fake_img1, opp_mask_p)
        fake_img2_final = fg_masked2 + bg_masked2 # Parent image
        parent_mask = fake_img2_mask
        masked_parent = fg_masked2

        #Child stage
        h_code3 = self.h_net3(h_code2, c_code)
        fake_img3_foreground = self.img_net3(h_code3) # Child foreground
        fake_img3_mask = self.img_net3_mask(h_code3) # Child mask
        ones_mask_c = torch.ones_like(fake_img3_mask)
        opp_mask_c = ones_mask_c - fake_img3_mask
        fg_masked3 = torch.mul(fake_img3_foreground, fake_img3_mask)
        bg_masked3 = torch.mul(fake_img2_final, opp_mask_c)
        img = fg_masked3 + bg_masked3  # Child image
        child_mask = fake_img3_mask
        masked_child = fg_masked3

        return img, bg_image, parent_mask, masked_parent, child_mask, masked_child


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class Generator():
    def __init__(self, cuda):
        if not cuda:
            raise NotImplementedError
        # initialise generator and load weights
        self.net = G_NET()
        self.net.cuda()
        self.net.apply(weights_init)
        self.net = torch.nn.DataParallel(self.net)
        model_dict = self.net.state_dict()
        state_dict = \
            torch.load(FINEGAN_WEIGHTS_PATH,
                       map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        self.net.load_state_dict(model_dict)
        # initialise distributions for latents
        self.c_dist = torch.distributions.OneHotCategorical(
            torch.ones(FINE_GRAINED_CATEGORIES).cuda(),
        )
        self.z_dist = torch.distributions.Normal(
            torch.tensor(0.).cuda(),
            torch.tensor(1.).cuda(),
        )
        self.net.eval()

    def __call__(self, B, category=None):
        """
        B: batch size
        """
        if category is None:
            c = self.c_dist.sample([B])
        else:
            c = torch.zeros(
                B, FINE_GRAINED_CATEGORIES
            ).float()
            c[:, category] = 1
        z = self.z_dist.sample([B, GAN_Z_DIM])
        return self.net(z, c)


def save_image(self, images, save_dir, iname):
    img_name = '%s.png' % (iname)
    full_path = os.path.join(save_dir, img_name)
    if (iname.find('mask') == -1) or (iname.find('foreground') != -1):
        img = images.add(1).div(2).mul(255).clamp(0, 255).byte()
        ndarr = img.permute(1, 2, 0).data.cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(full_path)
    else:
        img = images.mul(255).clamp(0, 255).byte()
        ndarr = img.data.cpu().numpy()
        ndarr = np.reshape(ndarr, (ndarr.shape[-1], ndarr.shape[-1], 1))
        ndarr = np.repeat(ndarr, 3, axis=2)
        im = Image.fromarray(ndarr)
        im.save(full_path)
