from os import mkdir
from os.path import join, isdir
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from mea.config import BIRD_IMG_DIM

def crop(pil_image, bbox):
    """
    from https://github.com/kkanshul/finegan/blob/master/code/datasets.py
    """
    width, height = pil_image.size
    r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    y1 = np.maximum(0, center_y - r)
    y2 = np.minimum(height, center_y + r)
    x1 = np.maximum(0, center_x - r)
    x2 = np.minimum(width, center_x + r)
    cropped = pil_image.crop([x1, y1, x2, y2])
    cropped.load()
    cropped = cropped.resize((BIRD_IMG_DIM,
                              BIRD_IMG_DIM),
                             resample=Image.LANCZOS)
    return cropped


UNPROCESSED = "data/birds/CUB_200_2011"
BBOX_FILE_PATH = join(UNPROCESSED, "bounding_boxes.txt")
IMAGES_FILE_PATH = join(UNPROCESSED, "images.txt")
IMAGES_DIR = join(UNPROCESSED, "images")
CLASSES_FILE = join(UNPROCESSED, "classes.txt")
ATTRIBUTES_FILE = join(UNPROCESSED, "attributes/image_attribute_labels.txt")
ATTRIBUTE_NAMES_FILE = join(UNPROCESSED, "attributes/attributes.txt")
TRAIN_SPLIT_FILE = join(UNPROCESSED, "train_test_split.txt")
ROOT_DIR = "data/birds/"   # where we save processed stuff to

# use classes file to create mapping from file name to class:
classes_dict = {}
classes = []
with open(CLASSES_FILE, 'r') as f:
    while True:
        line = f.readline()
        if line == '':
            break
        index, d = line.replace('\n', '').split(' ')
        classes_dict[d] = int(index) - 1
        classes.append(d.split('.')[1])
# print(classes)
def get_class(path):
    return classes_dict[path.split('/')[0]]

# create a mapping from attribute index to name
attributes_dict = {}
with open(ATTRIBUTE_NAMES_FILE, 'r') as f:
    while True:
        line = f.readline()
        if line == '':
            break
        index, name = line.replace('\n', '').split(' ')
        attributes_dict[int(index)-1] = name
    # print([attributes_dict[i] for i in range(312)] + ["class"])

# create set of training indices
train_set = set([])
with open(TRAIN_SPLIT_FILE, 'r') as f:
    while True:
        line = f.readline()
        if line == '':
            break
        index, is_train = line.replace('\n', '').split(' ')
        if int(is_train) == 1:
            train_set.add(int(index)-1)

# use attributes file to make a generator that iterates through attributes for each image
def iter_attributes():
    f = open(ATTRIBUTES_FILE, 'r')
    n_attributes = 312
    def readline():
        line = f.readline()
        line = line.replace('\n', '')
        line = line.replace('  ', '')
        if line == '':
            return None, None, None
        try:
            line = line.split(' ')
            image_i, attr, value, _, _ = line
        except ValueError:
            if line[0] in ('2275', '9364'):
                # Ignore extra column of zeros in these images
                image_i, attr, value = line[:3]
            else:
                print(line)
                raise ValueError
        if image_i == 100:
            return None, None, None
        return int(image_i)-1, \
            int(attr)-1, \
            int(value)
    current_image_i = 0
    image_i, attr, value = readline()
    while True:
        attributes = [-1] * 312
        while image_i == current_image_i:
            attributes[attr] = value
            image_i, attr, value = readline()
        # finalize and yield attributes for image
        assert -1 not in attributes
        yield current_image_i, attributes
        # update image index
        if image_i is None:
            return
        else:
            assert image_i == current_image_i + 1
            current_image_i = image_i

def iter_image_paths():
    image_paths_file = open(IMAGES_FILE_PATH, 'r').read().split('\n')
    for line in image_paths_file:
        if line == '':
            return
        index, path = line.split(' ')
        yield int(index) - 1, path

def iter_bboxes():
    bbox_file = open(BBOX_FILE_PATH, 'r').read().split('\n')
    for line in bbox_file:
        if line == '':
            return
        line = line.split(' ')
        assert len(line) == 5
        index = int(line[0]) - 1
        bbox = [float(f) for f in line[1:5]]
        yield index, bbox

# make directories for training and testing
TRAIN_DIR = join(ROOT_DIR, "train")
TEST_DIR = join(ROOT_DIR, "test")
TRAIN_IMAGE_DIR = join(TRAIN_DIR, "images")
TEST_IMAGE_DIR = join(TEST_DIR, "images")
TRAIN_ATTRIBUTES = join(TRAIN_DIR, "attributes.txt")
TEST_ATTRIBUTES = join(TEST_DIR, "attributes.txt")
for d in (TRAIN_DIR, TEST_DIR,
          TRAIN_IMAGE_DIR, TEST_IMAGE_DIR):
    if not isdir(d):
        mkdir(d)
for f in (TRAIN_ATTRIBUTES, TEST_ATTRIBUTES):
    open(f, 'w')
def NEW_IMAGE_PATH(index):
    D = TRAIN_IMAGE_DIR if index in train_set \
        else TEST_IMAGE_DIR
    return join(D, f"{index}.png")
def ATTRIBUTE_FILE(index):
    return TRAIN_ATTRIBUTES if index in train_set \
        else TEST_ATTRIBUTES

image_paths = iter_image_paths()
bboxes = iter_bboxes()
attrs = iter_attributes()
for (i1, path), \
    (i2, bbox), \
    (i3, attr) in zip(iter_image_paths(),
                      iter_bboxes(),
                      iter_attributes()):
    assert i1 == i2 and i2 == i3
    class_label = get_class(path)
    img = Image.open(join(IMAGES_DIR, path))
    cropped_img = crop(img, bbox)
    img_path = NEW_IMAGE_PATH(i1)
    cropped_img.save(img_path)
    annotations = attr + [class_label]
    txt = f"{img_path} {' '.join(map(str, annotations))}\n"
    attr_file = ATTRIBUTE_FILE(i1)
    open(attr_file, 'a').write(txt)
    if i1 % 100 == 0:
        print("done", i1)
