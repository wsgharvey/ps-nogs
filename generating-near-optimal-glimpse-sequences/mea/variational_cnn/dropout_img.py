import numpy as np
import torch
import torch.nn.functional as F

from ..utils import downsample, upsample
from ..config import IMG_DIM, ATT_DIM, \
    BIRD_ATT_DIM, BIRD_IMG_DIM

MIN_X = 0
MAX_X = IMG_DIM-ATT_DIM


def build_mask(shape,
               att_shape,
               locations=[]):
    mask = torch.zeros(shape)
    for r, c in locations:
        mask[r:r+att_shape[0],
             c:c+att_shape[1]] = 1
    return mask


def img_dropout_with_channel(x,
                             att_shape,
                             locations=[]):
    """
    image dropout on CPU for unbatched image.
    """
    mask = build_mask(x[0].shape,
                      att_shape=att_shape,
                      locations=locations)
    mask = mask.unsqueeze(0)
    return torch.cat((x*mask, mask), dim=0)


def batch_img_dropout_with_channel(xs,
                                   att_shape,
                                   locations=[]):
    """
    batched version of 'img_dropout_with_channel'
    if cuda=True, xs should already be on cuda.
    """
    B, _, H, W = xs.shape
    mask = build_mask((H, W),
                      att_shape,
                      locations)
    mask = mask.view(1, 1, H, W).expand(B, 1, H, W)
    mask = mask.to(xs.device)
    return torch.cat((xs*mask, mask), dim=1)
