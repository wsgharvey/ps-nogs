import numpy as np
import torch.utils.data as data
import torchvision.transforms as T

from ..dataset import CelebHQ, normalize
from ..config import TRAIN_INDICES, VALID_INDICES, \
    MAX_TRAINING_GLIMPSES, IMG_DIM, ATT_DIM, \
    CELEBHQ_MEAN, CELEBHQ_STD
from ..utils import set_random_seed
from .dropout_img import img_dropout_with_channel


def apply_random_attention(image):
    t = np.random.randint(1, MAX_TRAINING_GLIMPSES)
    locations = [(np.random.randint(0, IMG_DIM-ATT_DIM),
                  np.random.randint(0, IMG_DIM-ATT_DIM))
                 for _ in range(t)]
    return img_dropout_with_channel(image, (ATT_DIM, ATT_DIM),
                                    locations)


def apply_deterministic_attention(image):
    """
    makes locations attended to a deterministic function of the image.
    means that validation set is deterministic.
    """
    t_seed = image[:, -1].sum()
    t = 1 + int(t_seed*10000) % MAX_TRAINING_GLIMPSES

    def sample_loc(index):
        seed = image[:, index].sum()
        return int(seed*10000) % (IMG_DIM-ATT_DIM)
    locations = [(sample_loc(i), sample_loc(i+1))
                 for i in range(0, 2*t, 2)]
    return img_dropout_with_channel(image, (ATT_DIM, ATT_DIM),
                                    locations)


def worker_init_fn(i):
    """
    set seed for each worker to number generated my main seed
    """
    max_seed = 2**32
    seed = (np.random.randint(max_seed) + i) % max_seed
    set_random_seed(seed)


def get_loader(indices,
               batch_size,
               num_workers,
               random_horizontal_flip,
               dropout_transform,):
    data_transforms = []
    if random_horizontal_flip:
        data_transforms.append(T.RandomHorizontalFlip())
    data_transforms.append(T.ToTensor())
    data_transforms.append(normalize)
    data_transforms.append(dropout_transform)
    transform = T.Compose(data_transforms)
    dataset = CelebHQ(transform, indices)
    sampler = data.sampler.SubsetRandomSampler(indices)
    return data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn
    )


def get_train_valid_loaders(batch_size, n_workers):
    """
    For training variational CNN
    """
    # making two loaders uses more memory but is very simple...
    train_loader = get_loader(
        TRAIN_INDICES,
        batch_size=batch_size,
        num_workers=n_workers,
        random_horizontal_flip=True,
        dropout_transform=apply_random_attention,
    )
    val_loader = get_loader(
        VALID_INDICES,
        batch_size=int(1+batch_size/10),  # avoid weird out-of-memory crash
        num_workers=n_workers,
        random_horizontal_flip=False,
        dropout_transform=apply_deterministic_attention,
    )
    return train_loader, val_loader
