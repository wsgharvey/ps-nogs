import os
import git
import numpy as np
import torch
import random
import torch.nn.functional as F
from collections import OrderedDict

from mea.config import BIRD_IMG_DIM, BIRD_ATT_DIM, \
    MAX_TRAINING_GLIMPSES

def str2bool(string):
    return string.lower() == "true"

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, log_path, variables):
        self.log_path = log_path
        repo = git.Repo(search_parent_directories=True)
        self.git = repo.head.object.hexsha
        self.params = {k: v for k, v in variables.items()
                       if k.isupper()}
        self.valid_losses = []
        self.train_losses = []
        assert not os.path.isfile(log_path)
        with open(self.log_path, 'w') as f:
            f.write(self.git+'\n')
            for name, value in self.params.items():
                f.write(f'{name: <20} {value}\n')

    def add_epoch(self, epoch, train_loss, valid_loss):
        self.valid_losses.append(valid_loss)
        self.train_losses.append(train_loss)
        with open(self.log_path, 'a') as f:
            f.write(f'Train Loss: {train_loss:10.6f}    Validation Loss: {valid_loss:10.6f}\n')

    def got_best_valid_loss(self):
        return len(self.valid_losses) == 1 or self.valid_losses[-1] < min(self.valid_losses[:-1])

    def log_checkpoint(self, file_name):
        with open(self.log_path, 'a') as f:
            f.write(f'Checkpoint saved at {file_name}.\n')


def transformed_coords(old_shape, new_shape, old_coords):
    """
    gives coordinates for elements which were at `old_coords`
    before an object of shape `old_shape` was reshaped to `new_shape`
    old_shape, new_shape:  torch.Size objects
    old_coords:  tensor of dimensionality x num_coords
    """
    flat_coords = old_coords[:-1, :].T @ torch.cumprod(torch.tensor(old_shape[::-1]), 0).numpy()[:-1][::-1] \
                  + old_coords[-1, :]
    new_coords = []
    elements_per_layer = torch.cumprod(torch.tensor(new_shape[::-1]), 0).numpy()[:-1][::-1]
    for el in elements_per_layer:
        new_coords.append(flat_coords//el)
        flat_coords = flat_coords % el
    new_coords.append(flat_coords)
    return torch.tensor(new_coords)


def take_patch(img, img_size, pr, pc, psize):
    """
    extract a patch from img starting at (pr, pc) with size psize (same in each direction).
    Pad with zeros if not a good fit.
    """
    assert len(img.shape) == 3

    new_minr = max(0, -pr)
    new_maxr = min(psize, psize-pr)
    new_minc = max(0, -pc)
    new_maxc = min(psize, psize-pc)
    img_minr = max(0, pr)
    img_maxr = min(pr+img_size, img_size)
    img_minc = max(0, pc)
    img_maxc = min(pc+img_size, img_size)

    new_img = torch.zeros(img.shape[0], psize, psize)
    new_img[:, new_minr:new_maxr, new_minc:new_maxc] = img[:, img_minr:img_maxr, img_minc:img_maxc]
    return new_img


def bernoulli_entropy(p):
    """
    Batched entropy calculation for Bernoullis parameterised by `p`
    """
    r = 1-p
    entropy = -p*p.log()-r*r.log()
    # set nans to 0 (since 0*log(0) = 0 in entropy calculation)
    entropy[entropy != entropy] = 0
    return entropy

def categorical_entropy(probs):
    """
    Batched entropy calculation for categorical distribution.
    Assumes normalised (and non-log) probs of shape B x C.
    """
    entropy = -(probs*probs.log()).sum(dim=1)
    # set nans to 0 (since 0*log(0) = 0 in entropy calculation)
    entropy[entropy != entropy] = 0
    return entropy


def read_sequence(path):
    lines = open(path, 'r').readlines()
    return [[int(c) for c in line.split(', ')] for line in lines]


def write_sequence(path, observed_locations):
    with open(path, 'w') as f:
        for loc in observed_locations:
            f.write(
                ", ".join(str(l) for l in loc)\
                +'\n'
            )


class allow_unbatched(object):
    def __init__(self, input_correspondences):
        self.input_correspondences = \
            OrderedDict(input_correspondences)

    def __call__(self, f):
        def wrapped(*args, **kwargs):
            args = list(args)
            to_unbatch = []
            for inp_i, out_is in \
                    self.input_correspondences.items():
                inp = args[inp_i]
                assert len(inp.shape) in [3, 4]
                is_batched = len(inp.shape)==4
                if not is_batched:
                    args[inp_i] = inp.unsqueeze(0)
                    to_unbatch += out_is
            ret = f(*args, **kwargs)
            if type(ret) is not tuple:
                ret = (ret,)
            ret = list(ret)
            for out_i in to_unbatch:
                ret[out_i] = ret[out_i].squeeze(0)
            return tuple(ret) if len(ret) > 1 else ret[0]
        return wrapped


@allow_unbatched({0: [0]})
def upsample(x, new_size=None, scaling=None):
    if new_size is None:
        H = x.shape[2]
        assert H % scaling == 0
        new_size = H // scaling
    return F.interpolate(x,
                         (new_size, new_size),
                         mode='bilinear',
                         align_corners=False)

@allow_unbatched({0: [0]})
def downsample(x, new_size=None, scaling=None):
    if scaling is None:
        H = x.shape[2]
        assert H % new_size == 0
        scaling = H // new_size
    elif scaling == 1:
        return x
    return F.avg_pool2d(x, stride=scaling,
                        kernel_size=scaling)

@allow_unbatched({0: [0]})
def get_observed_patch(images, R, C, att_dim,
                       horizontal_flip=False):
    # find coordinates to take
    rows = torch.arange(R, R+att_dim)
    columns = torch.arange(C, C+att_dim)
    if horizontal_flip:
        width = images.shape[3]
        columns = width-1-columns

    # select pixels on GPU (if using it)
    rows = rows.to(images.device)
    columns = columns.to(images.device)
    return images\
        .index_select(2, rows)\
        .index_select(3, columns)

def sample_bird_glimpse_location(deterministic_seed=None):
    grid_dim = BIRD_IMG_DIM - BIRD_ATT_DIM
    if deterministic_seed is None:
        x, y = np.random.randint(0, grid_dim+1,
                                 size=2)
    else:
        if grid_dim == 0:
            x, y = 0, 0
        else:
            x = deterministic_seed % grid_dim
            deterministic_seed //= grid_dim
            y = deterministic_seed % grid_dim
    return x, y


def sample_bird_glimpse_sequence(end_at_T_1=False):
    T = MAX_TRAINING_GLIMPSES
    if end_at_T_1:
        T -= 1
    t = np.random.randint(1, T)
    return [sample_bird_glimpse_location()
            for _ in range(t)]
