from os import listdir
import glob
import torch.utils.data as data
import torchvision.transforms as transforms

from ..dataset import pil_loader, normalize
from ..config import FAKE_CELEBHQ_IMAGE_DIR, GET_FAKE_CELEBHQ_IMAGE_PATH, FAKE_CELEBHQ_N_WORKERS


class FakeCelebHQ(data.Dataset):
    def __init__(self, transform, optional_transform):
        self.transform = transform
        self.optional_transform = optional_transform
        self.length = len(glob.glob(GET_FAKE_CELEBHQ_IMAGE_PATH('*')))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # unpack index if necessary
        if self.optional_transform is not None:
            assert type(index) == tuple
            index, do_optional_transform = index
        else:
            do_optional_transform = False
        # load as PIL image
        sample = pil_loader(GET_FAKE_CELEBHQ_IMAGE_PATH(index))
        # apply transform(s)
        if do_optional_transform:
            sample = self.optional_transform(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def get_fake_loader(batch_size,
                    optional_flip):

    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    data_transforms.append(normalize)
    data_transforms = transforms.Compose(data_transforms)

    optional_transform = transforms.functional.hflip if optional_flip else None

    dataset = FakeCelebHQ(
        data_transforms,
        optional_transform=optional_transform,
    )

    return data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=FAKE_CELEBHQ_N_WORKERS,
        shuffle=False, pin_memory=True
    )


class BatchlessFakeLoader():
    """
    loads whatever is asked for, so batch size is irrelevant.
    """
    def __init__(self, optional_flip):
        self.loader = get_fake_loader(batch_size=100000,
                                      optional_flip=optional_flip)

    def load(self, indices, flipped=None):
        # check flipped is specified if and only if necessary
        assert (flipped is None) ^ \
            (self.loader.dataset.optional_transform is not None)

        self.loader.batch_sampler.sampler = \
            indices if flipped is None else zip(indices, flipped)
        return next(self.loader.__iter__())
