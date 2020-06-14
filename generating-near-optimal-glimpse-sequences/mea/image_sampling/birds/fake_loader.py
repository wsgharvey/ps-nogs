from os import listdir
from os.path import join
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import h5py

from mea.config import FAKE_BIRDS_PATH, FAKE_BIRDS_N_WORKERS
from mea.bird_dataset import normalize


class FakeBirds(data.Dataset):
    def __init__(self, image_type, transform, optional_flip,
                 N0=0, N=-1):
        """
        specifying N0 and N gives only data with indices between
        N0 and N0+N
        """
        self.transform = transform
        self.optional_flip = optional_flip
        self.N0 = N0
        self.N = N

        image_dir = join(FAKE_BIRDS_PATH, image_type)
        hdf5files = sorted(listdir(image_dir),
            key=lambda x: int(x.split('.')[0].split('-')[0]))

        # find ranges covered by each hdf5 file
        # and check that they are continuous from 0 to ...
        self.ranges = [tuple(int(i) for i in
                             fname.split('.')[0].split('-'))
          for fname in hdf5files]
        prev_high = 0
        for low, high in self.ranges:
            assert low == prev_high
            prev_high = high
        highs = [high for _, high in self.ranges]

        # construct mapping from index in entire dataset to
        # actual file
        def get_fname(index):
            for low, high in ranges:
                if high > index:
                    file_index = index - low
                    fname = f"{low}-{high}.hdf5"
                    return fname, index
            raise Exception(f"Index {index} not found.")

        # collate dataset names: do not load here as it will not
        # work without parallel h5py
        self.dataset_paths = [join(image_dir, fname)
                              for fname in hdf5files]

    def __len__(self):
        return self.ranges[-1][1] if self.N == -1 else self.N

    def __getitem__(self, index):
        # unpack optional flip bool if necessary
        if self.optional_flip:
            assert type(index) == tuple
            index, do_flip = index
        else:
            do_flip = False

        # add offset in case we are not using full dataset
        index = index + self.N0

        # load from hdf5 dataset
        image = None
        for dataset_no, \
                (low, high) in enumerate(self.ranges):
            if high > index:
                file_index = index - low
                dataset_path = self.dataset_paths[dataset_no]
                dataset = h5py.File(
                    dataset_path,
                    'r'
                )['data']
                image = dataset[file_index]
                break
        if image is None:
            raise Exception(f"Index {index} not we.")

        # found now have image as numpy array
        # shape (dim, dim, 3)
        # integers in [0, 255]
        mode = 'L' if image.shape[-1] == 1 else 'RGB'
        if mode == 'L':
            image = image[:, :, 0]
        if self.transform is not None:
            image = self.transform(image)
        if do_flip:
            assert image.shape[0] == 3 # check we are flipping right dim
            image = image.flip(dims=(2,))
        return image


def get_fake_loader(image_type,
                    batch_size,
                    pin_memory=True,
                    shuffle=False,
                    N0=0, N=-1,
                    optional_flip=False,
                    **kwargs):
    """
    Shamelessly copied and pasted from ..fake_loader
    """
    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    if image_type not in ['pm', 'cm']:
        data_transforms.append(normalize)
    data_transforms = transforms.Compose(data_transforms)

    dataset = FakeBirds(
        image_type,
        data_transforms,
        optional_flip=optional_flip,
        N0=N0, N=N
    )

    return data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=FAKE_BIRDS_N_WORKERS,
        shuffle=shuffle, pin_memory=pin_memory,
        **kwargs
    )


class BatchlessFakeLoader():
    """
    loads whatever is asked for, so batch size is irrelevant.
    """
    def __init__(self, image_type, optional_flip):
        self.loader = get_fake_loader(image_type,
                                      batch_size=100000,
                                      optional_flip=optional_flip,
                                      pin_memory=False)

    def load(self, indices, flipped=None):
        # check flipped is specified if and only if necessary
        assert (flipped is None) ^ \
            self.loader.dataset.optional_flip

        self.loader.batch_sampler.sampler = \
            indices if flipped is None else zip(indices, flipped)
        return next(self.loader.__iter__())
