import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
import math

from mea.config import BIRD_IMG_DIM, \
    BIRD_ATT_DIM, MAX_TRAINING_GLIMPSES
from mea.utils import set_random_seed, \
    sample_bird_glimpse_location
from mea.bird_dataset import Birds, normalize
from ..dropout_img import img_dropout_with_channel


def apply_random_attention(image):
    t = np.random.randint(1, MAX_TRAINING_GLIMPSES)
    locations = [sample_bird_glimpse_location()
                 for _ in range(t)]
    return img_dropout_with_channel(image,
                                    (BIRD_ATT_DIM, BIRD_ATT_DIM),
                                    locations)


def apply_deterministic_attention(image):
    """
    makes locations attended to a deterministic function of the image.
    an easier way to make validation deterministic than setting a random seed.
    """
    t_seed = image[:, -1].sum()
    t = 1 + int(t_seed*10000) % MAX_TRAINING_GLIMPSES
    def sample_loc(index):
        seed = int(image[:, index].sum()*10000)
        return sample_bird_glimpse_location(seed)
    locations = [sample_loc(i)
                 for i in range(0, t)]
    return img_dropout_with_channel(image,
                                    (BIRD_ATT_DIM, BIRD_ATT_DIM),
                                    locations)


def worker_init_fn(i):
    """
    set seed for each worker to number generated my main seed
    """
    max_seed = 2**32
    seed = (np.random.randint(max_seed) + i) % max_seed
    set_random_seed(seed)


def get_loader(attribute,
               mode,
               batch_size,
               num_workers,
               random_horizontal_flip,
               dropout_transform):
    data_transforms = []
    if random_horizontal_flip:
        data_transforms.append(T.RandomHorizontalFlip())
    data_transforms.append(T.ToTensor())
    data_transforms.append(normalize)
    data_transforms.append(dropout_transform)
    transform = T.Compose(data_transforms)
    dataset = Birds(transform, mode,
                    attribute)
    indices = list(range(len(dataset)))
    sampler = data.sampler.SubsetRandomSampler(indices)
    return data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn
    )


def get_train_valid_loaders(attribute, batch_size, n_workers):
    """
    For training variational CNN
    """
    # making two loaders uses more memory but is very simple...
    train_loader = get_loader(
        attribute,
        'train',
        batch_size=batch_size,
        num_workers=n_workers,
        random_horizontal_flip=True,
        dropout_transform=apply_random_attention,
    )
    val_loader = get_loader(
        attribute,
        'valid',
        batch_size=batch_size,
        num_workers=n_workers,
        random_horizontal_flip=False,
        dropout_transform=apply_deterministic_attention,
    )
    return train_loader, val_loader
