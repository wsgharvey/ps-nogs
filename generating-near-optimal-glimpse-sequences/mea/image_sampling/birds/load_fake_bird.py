import numpy
import torch
from os import listdir
from os.path import join
from shutil import copytree
import tarfile
import h5py

from PIL import Image
from mea.config import FAKE_BIRDS_PATH, BIRD_IMG_DIM

image_type = 'images'
image_dir = join(FAKE_BIRDS_PATH, image_type)

hdf5_files = listdir(image_dir)
ranges = sorted([tuple(int(i) for i in
                       fname.split('.')[0].split('-'))
          for fname in hdf5_files])

# check that ranges cover a continuous range from 0 to some number
prev_high = 0
for low, high in ranges:
    assert low == prev_high
    prev_high = high
highs = [high for _, high in ranges]
# make mapping from index to filename
def get_fname(index):
    for low, high in ranges:
        if high > index:
            file_index = index - low
            fname = f"{low}-{high}.hdf5"
            return fname, file_index
    raise Exception(f"Index {index} not found.")

# now get some files
for image_index in [23, 2344, 12000]:
    fname, findex = get_fname(image_index)
    dataset_path = join(image_dir, fname)
    dataset = h5py.File(dataset_path, 'r')['data']

    image = dataset[findex]
    Image.fromarray(image).save(f"retrieved_{image_index}.png")
