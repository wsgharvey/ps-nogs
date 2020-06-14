import argparse
from math import ceil
import time
import threading
import os
from os.path import join
from PIL import Image
from progress.bar import Bar
import torch
import numpy as np
import h5py

from mea.image_sampling.birds import Generator
from mea.config import FINEGAN_BATCH_SIZE, FAKE_BIRDS_PATH, \
    BIRD_IMG_DIM, BIRDS_DATASET_NAMES
from mea.utils import set_random_seed

parser = argparse.ArgumentParser(
    description='Generate structured fake bird images.'
)
parser.add_argument('--N0', type=int, default=0,
                    help='First index.')
parser.add_argument('--N', type=int,
                    help='Number of images to generate.')
args = parser.parse_args()

# Make the hdf5 datasets
for name in BIRDS_DATASET_NAMES:
    try:
        os.mkdir(join(FAKE_BIRDS_PATH, name))
    except FileExistsError:
        pass
are_masks = [False, False, True, False, True, False]
datasets = [h5py.File(join(FAKE_BIRDS_PATH, name,
                           f"{args.N0}-{args.N0+args.N}.hdf5"),
                      'w'
            ).create_dataset('data',
                             (args.N,
                              BIRD_IMG_DIM, BIRD_IMG_DIM,
                              1 if is_mask else 3,),
                             dtype=np.uint8)
            for name, is_mask in zip(BIRDS_DATASET_NAMES, are_masks)]

generator = Generator(cuda=True)

bar = Bar('Generating', max=ceil(args.N/FINEGAN_BATCH_SIZE))
start = time.time()
assert args.N0 % FINEGAN_BATCH_SIZE == 0, \
    f"N0 should be divisible by {BIRD_IMAGE_GEN_BATCH_SIZE} to \
      be reproducible if we rerun this."
for index in range(args.N0, args.N0+args.N, FINEGAN_BATCH_SIZE):
    set_random_seed(index)

    with torch.no_grad():
        image, bg_image, parent_mask, \
            masked_parent, child_mask, masked_child = \
                generator(FINEGAN_BATCH_SIZE)

    def image_to_numpy(img, is_mask):
        if not is_mask:
            img = img.add(1).div(2)
        img = img.mul(255).clamp(0, 255).byte()
        return img.permute(1, 2, 0).data.cpu().numpy()

    def save_all(index, image_batch,
                 is_mask, hdf5_dataset):
        for batch_index, image in enumerate(image_batch):
            hdf5_index = index-args.N0 + batch_index
            hdf5_dataset[hdf5_index] = \
                image_to_numpy(image, is_mask)

    threads = []
    for images, \
        is_mask, \
        dataset in zip([image, bg_image,
                        parent_mask, masked_parent,
                        child_mask, masked_child],
                       are_masks, datasets):
        t = threading.Thread(target=save_all,
                             args=[index, images,
                                   is_mask, dataset])
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    bar.suffix = f"({index-args.N0+FINEGAN_BATCH_SIZE}/{args.N}) \
            | Time: {time.time()-start:.1f}s | "
    bar.next()
bar.finish()
