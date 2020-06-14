from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from utils import plot_images
import torch
import os
import random
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
from itertools import islice

attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
              'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
              'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
              'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
              'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
              'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
              'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
              'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
              'Wearing_Necktie', 'Young']


def png_loader(filepath):
    return Image.open(filepath)


def npy_loader(filepath):
    image = np.load(filepath)
    image = Image.fromarray(np.rollaxis(image[0], 0, 3))
    return image


class CelebAHQ(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, sequences_dir, attr,
                 seq_transform, non_seq_transform, n_optimal_seqs, mode,
                 image_type, test_size):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.sequences_dir = sequences_dir
        self.attr = attr
        self.seq_transform = seq_transform
        self.non_seq_transform = non_seq_transform
        self.mode = mode
        self.n_optimal_seqs = n_optimal_seqs
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.image_type = image_type
        self.test_size = test_size
        if image_type == "png":
            self.loader = png_loader
            self.img_namer = lambda i: str(i).zfill(6) + '.png'
        elif image_type == "npy":
            self.loader = npy_loader
            self.img_namer = lambda i: 'imgHQ' + str(i).zfill(5) + '.npy'
        elif image_type == "debug":
            self.loader = lambda f: Image.fromarray(np.zeros((224, 224, 3)).astype(np.uint8))
            self.img_namer = lambda i: 'N/A'
        else:
            raise Exception(f"Given image type, {image_type}, not recognised.")
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        attr_dict = pickle.load(open(self.attr_path, "rb"))
        print(attr_dict.keys())
        files = np.arange(30000)

        # determine if optimal attention sequences are available for specified class
        if self.mode == "train":
            attr_index = attributes.index(self.attr)
            available_indices = listdir(self.sequences_dir)
            sequences_available = str(attr_index) in available_indices
        else:
            sequences_available = False

        # load optimal sequences for class if available
        if sequences_available:
            attr_seq_dir = join(self.sequences_dir, str(attr_index))
            def read_sequence(f_name):
                lines = open(join(attr_seq_dir, f_name), 'r').readlines()
                return torch.LongTensor([[int(c) for c in line.split(', ')] for line in lines])
            def read_index(f_name):
                return int(f_name.split('.')[0])
            seq_files = listdir(attr_seq_dir)
            if self.n_optimal_seqs is not None:
                assert len(seq_files) >= self.n_optimal_seqs
                seq_files = seq_files[:self.n_optimal_seqs]
            self.sequences = {read_index(f): read_sequence(f) for f in seq_files}
        else:
            print(f"Not using optimal attention sequences.")
            self.sequences = None

        true_counter = 0
        for i, file_no in enumerate(files):

            label = [attr_dict[self.attr][file_no] == 1]
            if label == [True]:
                true_counter += 1

            filename = self.img_namer(file_no)
            if i >= 30000 - self.test_size:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
        print('Finished preprocessing the CelebA-HQ dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = self.loader(os.path.join(self.image_dir, filename))
        label = torch.LongTensor(label)
        has_target = torch.tensor(
            1 if self.sequences is not None and index in self.sequences else 0
        ).type(torch.LongTensor)
        target = self.sequences[index] if has_target else torch.zeros(5, 2).type(torch.LongTensor)
        transform = self.seq_transform if has_target else self.non_seq_transform
        img = transform(image).type(torch.FloatTensor)
        return img, label, has_target, target

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class FixedPropTrainBatchSampler(torch.utils.data.Sampler):
    """
    indices -            all indices
    supervised_indices - subset of indices with supervision
    """
    def __init__(self, indices, supervised_indices, supervised_prop, anneal_epochs):
        self.sup_indices = supervised_indices
        self.unsup_indices = [i for i in indices if i not in set(supervised_indices)]
        self.init_supervised_prop = supervised_prop
        self.anneal_rate = 0
        self.supervised_prop = supervised_prop

    def _anneal(self):
        self.supervised_prop -= self.anneal_rate
        if self.supervised_prop <= 0:
            self.supervised_prop = 0
            self.unsup_indices = self.unsup_indices + self.sup_indices
            self.sup_indices = []
            return True
        return False

    def __len__(self):
        return len(self.sup_indices) + len(self.unsup_indices)

    def __iter__(self):
        def iterate_shuffled(indices):
            indices_ = indices
            while True:
                random.shuffle(indices_)
                for i in indices_:
                    yield i
        unsup = iterate_shuffled(self.unsup_indices)
        sup = iterate_shuffled(self.sup_indices)
        sup_given, unsup_given = 1, 1  # non-zero to prevend DivisionByZeroError
        for _ in range(len(self)):
            if sup_given/(sup_given+unsup_given) < self.supervised_prop:
                yield next(sup)
                sup_given += 1
            else:
                yield next(unsup)
                unsup_given += 1


def get_train_celebhq_loader(image_dir,
                             attr_path,
                             sequences_dir,
                             attr,
                             n_optimal_seqs,
                             fixed_supervised_proportion,
                             anneal_epochs,
                             crop_size,
                             image_size,
                             batch_size,
                             mode,
                             num_workers,
                             valid_size,
                             test_size,
                             show_sample,
                             grayscale,
                             image_type,
                             ):
    """Build and return a data loader."""
    train_size = 27000
    valid_size = 500
    test_size = 2500

    if image_dir == 'default':
        image_dir = join(os.environ["SLURM_TMPDIR"], 'images')

    transform = []
    if grayscale:
        transform.append(T.Grayscale(num_output_channels=1))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    channels = 1 if grayscale else 3
    transform.append(T.Normalize(mean=(0.5,)*channels,
                                 std=(0.5,)*channels))
    transform = T.Compose(transform)
    seq_transform = transform
    non_seq_transform = T.Compose([T.RandomHorizontalFlip(), transform]) if mode == 'train' else transform
    dataset = CelebAHQ(image_dir, attr_path, sequences_dir, attr,
                       seq_transform, non_seq_transform, n_optimal_seqs, mode,
                       image_type, test_size)

    if mode == "train":
        train_idx = list(range(0, train_size))
        valid_idx = list(range(train_size, train_size+valid_size))
        test_idx = [1, 2, 3]  # not used
    elif mode == "test":
        train_idx = [1, 2, 3]
        valid_idx = [1, 2, 3]
        test_idx = list(range(test_size))

    if fixed_supervised_proportion is not None:
        train_sampler = FixedPropTrainBatchSampler(train_idx,
                                                   list(dataset.sequences.keys()),
                                                   fixed_supervised_proportion,
                                                   anneal_epochs)
    else:
        assert anneal_epochs is None
        train_sampler = SubsetRandomSampler(train_idx)

    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  num_workers=8)

    valid_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  drop_last=False,
                                  num_workers=8)

    test_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  sampler=test_sampler,
                                  drop_last=False,
                                  num_workers=8)

    if False:
        sample_loader = torch.utils.data.DataLoader(
            dataset, batch_size=9, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader, test_loader)

def anneal_hq_loader(loader):
    if not hasattr(loader.sampler, "_anneal"):
        return
    finished_annealing = loader.sampler._anneal()
    if finished_annealing:
        loader.dataset.sequences = None
