import numpy as np
from scipy.ndimage.morphology import binary_erosion
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from mea.utils import downsample, upsample, allow_unbatched
from mea.variational_cnn.dropout_img import build_mask
from mea.config import BIRD_IMG_DIM, BIRD_ATT_DIM, \
    BIRD_ATT_SCALES

@allow_unbatched({0: [0]})
def erode(binary_image, erosion=1):
    """
    Sets 1s at boundaries of binary_image to 0
    """
    batch_array = binary_image.data.cpu().numpy()
    return torch.tensor(
        np.stack([
            binary_erosion(
                array,
                iterations=erosion,
                border_value=1,  # so that we don't get border of zeros
            ).astype(array.dtype)
            for array in batch_array])
    ).to(binary_image.device)

def ConvBNReLU(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c,
                  kernel_size=3,
                  padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

def ConvPool(in_c, out_c):
    return nn.Sequential(
        ConvBNReLU(in_c, in_c),
        ConvBNReLU(in_c, out_c),
        nn.MaxPool2d(2, 2),
    )

class Discriminator(nn.Module):
    """
    Discriminates between images with sampled patches and then
    true and false completions.

    Extensions:
      - get some factorisation by separately scoring the foregrounds and backgrounds
        with two different discriminators
      - do batched image dropout stuff
    """
    def __init__(self, emb_erosion=2, base_channels=64):
        super().__init__()
        self.erosion = emb_erosion
        in_c = len(BIRD_ATT_SCALES)*4
        c = base_channels
        self.classifier = nn.Sequential(
            ConvPool(in_c, c),      # 64x64
            ConvPool(c, c),         # 32x32
            ConvPool(c, 2*c),       # 16x16
            ConvPool(2*c, 2*c),     # 8x8
            ConvPool(2*c, 4*c),     # 4x4
            ConvPool(4*c, 8*c),     # 2x2
            nn.Flatten(),
            nn.Linear(4*8*c, 1),
            nn.Sigmoid(),
        )

    @allow_unbatched({1: [0], 2: []})
    def fake_embedder(self, glimpsed_images,
                      completion_images,
                      locations,
                      erosion=1):
        """
        images can be batched, locations cannot be
        """
        locations = [(scale, [(x, y) for x, y, s in
                              locations if s == scale])
                     for scale in BIRD_ATT_SCALES]
        inputs = []
        for scale, coords in locations:
            completion = downsample(completion_images,
                                    scaling=scale)
            if len(coords) == 0:
                morphed = torch.cat(
                    [completion,
                     completion[:, :1]*0.],  # mask channel
                    dim=1)
            else:
                B, _, H, W = completion.shape
                glimpsed = downsample(glimpsed_images,
                                      scaling=scale)
                scaled_coords = [(r//scale, c//scale)
                                 for r, c in coords]
                obs_mask = build_mask(
                    shape=(H, W),
                    att_shape=(BIRD_ATT_DIM,
                               BIRD_ATT_DIM),
                    locations=scaled_coords
                ).unsqueeze(0)\
                 .unsqueeze(0)\
                 .to(glimpsed.device)  # batch + channel dims
                # erode each masks to make some zeros:
                eroded_obs_mask = erode(obs_mask,
                                        erosion)
                eroded_completion_mask = erode(1-obs_mask,
                                               self.erosion)
                morphed = (completion *\
                           eroded_completion_mask) + \
                          (glimpsed *\
                           eroded_obs_mask)
                morphed = torch.cat([
                    morphed,
                    obs_mask.expand(B, -1, -1, -1)],
                                    dim=1)
            inputs.append(
                upsample(morphed, new_size=BIRD_IMG_DIM)
            )
        final = torch.cat(inputs, dim=1)
        return final

    def real_embedder(self, image, locations):
        return self.fake_embedder(image, image,
                                  locations)

    def forward(self, glimpsed_images,
                completion_images,
                locations, single_seq=False):
        """
        Outputs distribution over completed_image being
        completion for glimpsed_image given observed
        locations.
        """
        if single_seq:
            embeddings = self.fake_embedder(
                glimpsed_images,
                completion_images,
                locations
            )
        else:
            embeddings = torch.stack(
                [self.fake_embedder(
                    glimpsed, completion, locs
                )
                 for glimpsed, completion, locs
                 in zip(glimpsed_images, completion_images,
                        locations)],
                dim=0
            )

        # # save some embeddings
        # import torchvision.transforms as T
        # for i, embedding in enumerate(embeddings):
        #     for c, channel in enumerate(embedding):
        #         image = channel.unsqueeze(0) * 0.12 + 0.44
        #         T.ToPILImage()(image).save(f"disc_emb_{i}_{c}.png")

        y = self.classifier(embeddings)
        return y

