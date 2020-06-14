import itertools as it
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

import numpy as np
from location import Location
from utils import str2bool, get_flops


def foveate(x, locs, glimpse_size):
    glimpses = []
    for img, coord in zip(x, locs.pixel):
        patch_r, patch_c = coord
        from_r = patch_r.item()
        from_c = patch_c.item()
        to_r = from_r + glimpse_size
        to_c = from_c + glimpse_size
        glimpses.append(img[:, from_r:to_r, from_c:to_c])
    return torch.stack(glimpses, dim=0)

class ConvBNReLU(nn.Sequential):
    """
    from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/mobilenet.py
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size,
                      stride, padding, bias=False, groups=groups),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class DepthwiseConv(nn.Conv2d):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        super().__init__(in_planes, out_planes, kernel_size,
                         stride=stride, padding=padding, groups=in_planes)

class PointwiseConv(nn.Conv2d):
    def __init__(self, in_planes, out_planes):
        super().__init__(in_planes, out_planes, kernel_size=1)

class SeparatedConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, hidden_planes,
                 kernel_size, stride=1):
        super().__init__(
            PointwiseConv(in_planes, hidden_planes),
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU(inplace=True),
            DepthwiseConv(hidden_planes, hidden_planes,
                          kernel_size, stride=stride),
            PointwiseConv(hidden_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

class ShuffleConv(nn.Module):
    def __init__(self, in_planes, out_planes,
                 kernel_size, stride=1):
        super().__init__()
        if in_planes > 16:
            self.groups = 4
        elif in_planes > 8:
            self.groups = 2
        else:
            self.groups = 1
        self.out_channels_per_group = out_planes // self.groups
        self.conv = ConvBNReLU(in_planes, out_planes, kernel_size,
                               stride=stride, groups=self.groups)

    def forward(self, x):
        x = self.conv(x)
        # now do channel shuffle
        B, C, H, W = x.shape
        x = x.view(B, self.groups, self.out_channels_per_group, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, C, H, W)
        return x

class ShuffleConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes,
                 kernel_size, stride=1):
        super().__init__(
            ShuffleConv(in_planes, out_planes,
                        kernel_size, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )


def get_glimpse_net(blocks, net_type, out_dim):
    # old architecture \approx standard_fc_True
    # conv_depth \approx 2-32-3-64-0-64
    convs16x16, channels16x16,\
        convs8x8, channels8x8,\
        convs4x4, channels4x4 = map(int, blocks.split('-'))
    # net_type is shuffle, separated or standard, avg or fc, True or False
    conv_type, end_with, initial5x5 = net_type.split('_')
    initial5x5 = str2bool(initial5x5)

    prev_channels = 3
    blocks = []
    dim = 16
    max_tensor_size = 3*16*16
    for dim, n_convs, n_channels in [(16, convs16x16, channels16x16),
                                     (8, convs8x8, channels8x8),
                                     (4, convs4x4, channels4x4),]:
        for conv in range(n_convs):
            stride = (2 if conv == 0 and dim != 16 else 1)
            if conv_type == 'standard' or len(blocks) == 0:
                conv = ConvBNReLU(prev_channels, n_channels,
                                  kernel_size=5 if initial5x5 and len(blocks) == 0 else 3,
                                  stride=stride,
                )
            elif conv_type == 'separated':
                conv = SeparatedConvBNReLU(prev_channels, n_channels, n_channels,
                                           kernel_size=3,
                                           stride=stride,
                )
            elif conv_type == 'shuffle':
                conv = ShuffleConvBNReLU(prev_channels, n_channels,
                                         kernel_size=3,
                                         stride=stride,
                )
            else:
                raise Exception("Bad conv_type", conv_type)
            blocks.append(
                conv
            )
            prev_channels = n_channels
            dim //= stride
            max_tensor_size = max(max_tensor_size, n_channels*dim**2)
        if n_convs == 0 and dim != 16:
            blocks.append(nn.MaxPool2d(2, 2))    # use max pool if we didn't get a chance to do strided conv
            dim //= 2

    # add final average pool / fully connected layer
    if end_with == 'avg':
        blocks.extend([nn.AvgPool2d(4),
                       nn.Flatten(),
                       nn.Linear(channels4x4, out_dim),
                       nn.BatchNorm1d(out_dim)])
    elif end_with == 'linear':
        blocks.extend([nn.Flatten(),
                       nn.Linear(4*4*channels4x4, out_dim),
                       nn.BatchNorm1d(out_dim)])
    else:
        assert end_with == 'fc'
        blocks.extend([nn.Flatten(),
                       nn.Linear(4*4*channels4x4, 128),
                       nn.ReLU(),
                       nn.Linear(128, out_dim),
                       nn.BatchNorm1d(out_dim)])


    return nn.Sequential(*blocks), max_tensor_size


class discrete_location_network(nn.Module):
    """
    like `location_network` but outputs categorical distribution over
    `output_size` locations
    """
    def __init__(self, input_size, output_size):
        super(discrete_location_network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, h_t):
        return self.fc(h_t)

    def get_loc_index_dist(self, x):
        return torch.distributions.Categorical(self(x))


class mixture_location_network(nn.Module):
    """
    samples location from a Gaussian mixture model
    """
    def __init__(self, input_size, n_components, min_std):
        super().__init__()
        self.fc = nn.Linear(input_size, n_components*(1+2+3))  # 1 probs, 2 mean, 3 covar
        self.n = n_components
        self.min_std = min_std
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x):
        # only used to calculate FLOPs
        B = x.shape[0]
        params = self.fc(x).view(B, -1, self.n)
        mixture_probs = F.log_softmax(params[:, 0], dim=1)     # B x n
        means = -1 + 2*self.sigmoid(params[:, 1:3])            # B x 2 x n
        scale_tril = torch.zeros(B, 2, 2, self.n)
        scale_tril[:, 0, 0] = self.softplus(params[:, 3]) + self.min_std
        scale_tril[:, 1, 1] = self.softplus(params[:, 4]) + self.min_std
        scale_tril[:, 1, 0] = params[:, 5]
        return params

    def get_loc_index_dist(self, x):
        B = x.shape[0]
        # parameterize mixture model in [-1, 1]^2
        params = self.fc(x).view(B, self.n, -1)
        mixture_probs = F.log_softmax(params[:, :, 0], dim=1)                  # B x n
        means = -1 + 2*self.sigmoid(params[:, :, 1:3])                         # B x n x 2
        cov = torch.zeros(B, self.n, 2, 2)                              # B x n x 2 x 2
        v1 = self.softplus(params[:, :, 3])
        v2 = self.softplus(params[:, :, 4])
        correlation = -1 + 2*self.sigmoid(params[:, :, 5])
        max_covariance = torch.sqrt(v1 * v2)
        cov[:, :, 0, 0] = v1 + self.min_std
        cov[:, :, 1, 1] = v2 + self.min_std
        cov[:, :, 1, 0] = correlation*max_covariance
        cov[:, :, 0, 1] = correlation*max_covariance

        # calculate discrete probabilities of points in 50x50 grid in [-1, 1]^2
        n_locs = 2500
        grid = Location(torch.arange(n_locs), 'index').normalized        # 2500 x 2
        grid = grid.view(n_locs, 1, 1, 2).expand(n_locs, B, self.n, 2)   # 2500 x B x n x 2
        logits = torch.distributions.MultivariateNormal(
            means,
            covariance_matrix=cov,
        ).log_prob(grid)
        logits = logits + mixture_probs                 # 2500 x B x n
        logits = torch.logsumexp(logits, dim=-1)        # 2500 x B
        logits = logits.transpose(1, 0)

        # construct categorical and return sample
        dist = torch.distributions.Categorical(logits=logits)
        return dist

def get_classifier(classifier_size, hidden_dim, num_classes):
    modules = []
    current_dim = hidden_dim
    for hidden_dim in classifier_size:
        modules.append(nn.Linear(current_dim, hidden_dim))
        modules.append(nn.ReLU())
        current_dim = hidden_dim
    modules.append(nn.Linear(current_dim, num_classes))
    modules.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*modules)


class HardAttentionNetwork(nn.Module):
    def __init__(self, T, hidden_dim, conv_depth, conv_type,
                 classifier_size, num_classes, glimpse_size,
                 loc_dist_type, loc_mixture_components,
                 loc_std, heuristic_dist_path):
        assert loc_dist_type in ['categorical', 'mixture']
        super().__init__()
        self.T = T
        self.glimpse_size = glimpse_size
        self.num_classes = num_classes
        if conv_type == 'old':
            self.glimpse_embedder = old_glimpse_network()
            self.cnn_max_tensor_size = 16384
            self.loc_embedder = old_loc_network()
        else:
            self.glimpse_embedder, self.cnn_max_tensor_size = get_glimpse_net(conv_depth,
                                                                              conv_type,
                                                                              hidden_dim)
            self.loc_embedder = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.baseline_net = nn.Linear(hidden_dim, 1)
        if loc_dist_type == 'mixture':
            self.location_net = mixture_location_network(hidden_dim,
                                                         loc_mixture_components,
                                                         loc_std)
        else:
            self.location_net = discrete_location_network(hidden_dim, 2500)
        self.classifiers = nn.ModuleList(
            [None]*(T-1) + \
            [get_classifier(classifier_size,
                            hidden_dim, num_classes)]
        )
        self.init_hidden = nn.Parameter(torch.randn(1, hidden_dim))
        self.hidden = None

    def initialize(self, batch_size):
        self.hidden = self.init_hidden.clone().expand(batch_size, -1)

    def get_loc_index_dist(self):
        return self.location_net.get_loc_index_dist(self.hidden.detach())

    def take_glimpse(self, x, loc):
        glimpse = foveate(x, loc, self.glimpse_size)
        y_emb = self.glimpse_embedder(glimpse).squeeze(-1).squeeze(-1)
        l_emb = self.loc_embedder(loc.normalized)
        yl_emb = F.relu(y_emb + l_emb)
        self.hidden = self.rnn(yl_emb, self.hidden)

    def get_baseline(self):
        return self.baseline_net(self.hidden.detach())

    def classify(self, t):
        # should be t of previous time step
        return self.classifiers[t](self.hidden)

    def step(self, t, x, replace_l_t, replacement_l_t):
        # sample locations and replace those that have supervision targets
        loc_index_dist = self.get_loc_index_dist()
        sampled_loc_indices = loc_index_dist.sample()
        if replace_l_t is None:
            loc_indices = sampled_loc_indices
        else:
            replacement_indices = replacement_l_t.index
            loc_indices = torch.mul(sampled_loc_indices, 1-replace_l_t) + \
                          torch.mul(replacement_indices, replace_l_t)
        loc = Location(loc_indices, 'index')
        # take glimpse and update hidden state
        self.take_glimpse(x, loc)
        return loc, loc_index_dist

    def get_EIG(self, t):
        return self.EIG_nets[t](self.hidden.detach()).view(-1)

    def forward(self, returns, x, replace_l=None, replacement_l=None):
        """
        returns: list of strings specifying what to return. Allowed strings are:
            ["prediction_probs",  (returns B x T x n_classes tensor)
             "locations",    (returns B x T Location)
             "baselines",    (returns B x T tensor)
             "selected_actions_prob", (returns B x T tensor)
             "entropies", (returns B x T tensor)
             "EIGs", (returns B x T-1 tensor)]
        Will return a dict with these as the keys.
        x: images to process
        - replace_l_t: B LongTensor of 1s and 0s for whether to replace each
                       proposed location with optimal targets
        - replacement_l_t: B x T optimal location targets - can be arbitrary
                           wherever replace_l_t is 0
        """
        returns = {k: [] for k in returns}

        # initalize hidden state
        self.initialize(x.shape[0])

        for t in range(self.T):

            if "baselines" in returns:
                b_t = self.get_baseline().squeeze()
                returns["baselines"].append(b_t)

            # take glimpse
            replacement_l_t = None if replace_l is None else replacement_l[:, t]
            l_t, l_t_dist = self.step(t, x, replace_l, replacement_l_t)

            if "locations" in returns:
                returns["locations"].append(l_t)
            if "selected_actions_prob" in returns:
                returns["selected_actions_prob"].append(l_t_dist.log_prob(l_t.index))
            if "entropies" in returns:
                returns["entropies"].append(l_t_dist.entropy())
            if "prediction_probs" in returns:
                if t == self.T-1:
                    probs = self.classify(t)
                else:
                    probs = torch.zeros(x.shape[0], self.num_classes)
                returns["prediction_probs"].append(probs)


        # format all the returns properly
        if "locations" in returns:
            location_indices = [locs.index for locs in returns["locations"]]
            returns["locations"] = Location(torch.stack(location_indices, dim=1), "index")
        if "baselines" in returns:
            returns["baselines"] = torch.stack(returns["baselines"], dim=1)
        if "selected_actions_prob" in returns:
            returns["selected_actions_prob" ] = torch.stack(returns["selected_actions_prob"], dim=1)
        if "prediction_probs" in returns:
            returns["prediction_probs"] = torch.stack(returns["prediction_probs"], dim=1)
        if "entropies" in returns:
            returns["entropies"] = torch.stack(returns["entropies"], dim=1)
        return returns
