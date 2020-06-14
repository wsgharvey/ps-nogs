from os.path import join
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .config import CELEBHQ_ANNOTATION_FILE, CELEBHQ_IMAGE_PATH, CELEBHQ_MEAN, CELEBHQ_STD


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        try:
            return img.convert('RGB')
        except:
            print(path)


class CelebHQ(Dataset):
    def __init__(self, transform, indices):
        self.images = []
        self.targets = []
        for line in open(join(CELEBHQ_ANNOTATION_FILE), 'r'):
            sample = line.split()
            assert len(sample) == 41, \
                f"Expected 40 labels and image name. Got f{len(sample)} elements."
            self.images.append(join(CELEBHQ_IMAGE_PATH, sample[0][:-3]+'png'))
            self.targets.append([1 if int(i) == 1 else 0 for i in sample[1:]])
        assert len(self.images) == 30000, \
            f"Expected 30000 data points in annotations file, got {len(self.images)}."
        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        assert index in self.indices  # ensure train loader doesn't access test data etc.
        path = self.images[index]
        image = self.transform(pil_loader(path))
        target = torch.LongTensor(self.targets[index])
        return image, target

    def __len__(self):
        return len(self.images)


normalize = T.Normalize(mean=CELEBHQ_MEAN,
                        std=CELEBHQ_STD)
to_normalized_tensor = T.Compose([T.ToTensor(), normalize])
