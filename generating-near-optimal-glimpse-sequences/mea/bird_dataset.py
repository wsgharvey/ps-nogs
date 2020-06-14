import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

from .config import BIRDS_MEAN, BIRDS_STD, \
    BIRDS_TRAIN_ANNOTATIONS, BIRDS_TEST_ANNOTATIONS, \
    birds_train, birds_valid, birds_test, \
    bird_valid_indices, bird_train_indices
from .dataset import pil_loader

normalize = T.Normalize(mean=BIRDS_MEAN,
                        std=BIRDS_STD)
to_normalized_tensor = T.Compose([T.ToTensor(), normalize])


class Birds(Dataset):
    def __init__(self, transform, mode, attribute):
        """
        Load annotations file. Process to get list of file paths and list
        of desired attribute.

        attribute is either attribute index (as defined in bird_attributes.py)
        or 'class' to classify bird type or 'all' to load all
        """
        assert mode in ['train', 'valid', 'test']
        self.transform = transform
        self.image_paths = []
        self.labels = []
        if attribute == 'class':
            attribute = -1

        # load stuff from annotations file
        annotations_path = BIRDS_TEST_ANNOTATIONS if mode == 'test' \
            else BIRDS_TRAIN_ANNOTATIONS
        for line in open(annotations_path, 'r'):
            datum = line.replace('\n', '').split(' ')
            image_path = datum[0]
            attributes = datum[1:]
            if attribute == "all":
                label = torch.LongTensor([int(a) for a in attributes])
            else:
                label = torch.tensor(int(attributes[attribute])).long()
            self.image_paths.append(image_path)
            self.labels.append(label)

        # chuck out training/valid annotations if we are doing
        # valid/train
        if mode == 'valid':
            self.image_paths = [self.image_paths[v]
                                for v in bird_valid_indices]
            self.labels = [self.labels[v] for v
                           in bird_valid_indices]
        elif mode == 'train':
            self.image_paths = [self.image_paths[v]
                                for v in bird_train_indices]
            self.labels = [self.labels[v] for v
                           in bird_train_indices]

        # check we have all the images/labels we expect
        assert len(self.image_paths) == len(self.labels)
        if mode == 'train':
            assert len(self.image_paths) == 4994
        elif mode == 'valid':
            assert len(self.image_paths) == 1000
        else:
            assert len(self.image_paths) == 5794

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = self.transform(pil_loader(path))
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.image_paths)

