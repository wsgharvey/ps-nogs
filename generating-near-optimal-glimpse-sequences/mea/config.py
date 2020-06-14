from os.path import join
import numpy as np
from os import environ

# CelebHQ train/valid/test splits--------------
"""
We will not touch first 2000 images until making
optimal sequences for them. Also leave last 2500
alone for use in validation/testing of final
partially supervised attention mechanism.
"""
do_oed = 2000
train = 25000
valid = 500
test = 2500
assert do_oed + train + valid + test == 30000
OED_INDICES = list(range(do_oed))
TRAIN_INDICES = list(range(do_oed, do_oed+train))
VALID_INDICES = list(range(do_oed+train, do_oed+train+valid))
TEST_INDICES = list(range(do_oed+train+valid, do_oed+train+valid+test))

# CelebHQ train/valid/test splits--------------
birds_valid = 1000   # we will also use these for the OED
birds_train = 4994
bird_valid_indices = [int(i) for i in
                      np.linspace(0, 5993, 1000)]
bird_train_indices = [i for i in range(birds_train+birds_valid)
                      if i not in bird_valid_indices]
assert len(bird_train_indices) + \
    len(bird_valid_indices) == 5994
birds_test = 5794

# data paths-----------------------------------
# Celeb-HQ
CELEBHQ_PATH = "../data/celebhq/"
CELEBHQ_IMAGE_PATH = join(CELEBHQ_PATH, "images/")
CELEBHQ_ANNOTATION_FILE = join(CELEBHQ_PATH, "annotations.txt")
FAKE_CELEBHQ_PATH = "../data/fake-celebhq/"
FAKE_CELEBHQ_IMAGE_DIR = join(FAKE_CELEBHQ_PATH, "images")
def GET_FAKE_CELEBHQ_IMAGE_PATH(index): return join(FAKE_CELEBHQ_IMAGE_DIR, f'{index}.png')
FAKE_CELEBHQ_PCA_PATH = join(FAKE_CELEBHQ_PATH, "pca/")
STYLEGAN_WEIGHTS_PATH = "mea/image_sampling/stylegan-weights/stylegan.pt"   # also hardcoded in get-stylegan-weights.sh
# Birds
BIRDS_PATH = "../data/birds/"
FAKE_BIRDS_PATH = "../data/fake-birds"
FAKE_BIRDS_PCA_PATH = join(FAKE_BIRDS_PATH, 'pca')
BIRDS_TRAIN_PATH = join(BIRDS_PATH, 'train')
BIRDS_TEST_PATH = join(BIRDS_PATH, 'test')
BIRDS_TRAIN_ANNOTATIONS = join(BIRDS_TRAIN_PATH,  "attributes.txt")
BIRDS_TEST_ANNOTATIONS = join(BIRDS_TEST_PATH,  "attributes.txt")
BIRDS_DATASET_NAMES = ['images', 'backgrounds', 'pm', 'mp', 'cm', 'mc']
FINEGAN_WEIGHTS_PATH = "finegan-weights/birds.pt"
# Other
CNN_SAVE_PATH = "trained-nets/"
CNN_LOG_PATH = "logs/"
OPTIMAL_SEQUENCE_DIR = "../optimal-sequences/celebhq/"
def GET_OPTIMAL_SEQUENCE_ATTR_DIR(attr): return join(OPTIMAL_SEQUENCE_DIR, str(attr))
def GET_OPTIMAL_SEQUENCE_PATH(attr, index): return join(GET_OPTIMAL_SEQUENCE_ATTR_DIR(attr), f"{index}.txt")
OED_LOG_DIR = "oed-logs/"
def GET_OED_LOG_ATTR_DIR(attr): return join(OED_LOG_DIR, str(attr))
def GET_OED_FIRST_LOC_PATH(attr, seed, net_path, mod_time):
    return join(GET_OED_LOG_ATTR_DIR(attr), f"first-loc_seed-{seed}_{net_path.replace('/', '--')}_{mod_time}.txt")
def GET_ENTROPY_MAP_PATH(attr, index): return join(GET_OED_LOG_ATTR_DIR(attr), f"EPE_{index}.p")
def GET_IMAGE_SAMPLE_LOG_PATH(attr, index): return join(GET_OED_LOG_ATTR_DIR(attr), f"samples_{index}.p")
PLOTTING_DIR = "figures/"
# Bird 'Other'
BIRDS_CNN_SAVE_PATH = "trained-nets/birds/"
BIRDS_CNN_LOG_PATH = "logs/birds/"
BIRDS_OPTIMAL_SEQUENCE_DIR = "../optimal-sequences/birds/"
BIRDS_OED_LOG_DIR = "bird-oed-logs/"
def GET_BIRDS_OPTIMAL_SEQUENCE_PATH(index):
    return join(BIRDS_OPTIMAL_SEQUENCE_DIR, f"{index}.txt")
def GET_BIRDS_OED_FIRST_LOC_PATH(seed, net_path, mod_time):
    return join(BIRDS_OED_LOG_DIR, f"first-loc_seed-{seed}_{net_path.replace('/', '--')}_{mod_time}.txt")
def GET_BIRDS_ENTROPY_MAP_PATH(index):
    return join(BIRDS_OED_LOG_DIR, f"EPE_{index}.p")
def GET_BIRDS_IMAGE_SAMPLE_LOG_PATH(index):
    return join(BIRDS_OED_LOG_DIR, f"samples_{index}.p")

# attention parameters------------------------
IMG_DIM = 224
ATT_DIM = 16
MAX_GLIMPSES = 1  # TODO change back to 5
LOC_GRID_SIZE = 50
POSSIBLE_LOCATIONS = [(int(x), int(y))
                      for x in np.linspace(0, IMG_DIM-ATT_DIM-1, LOC_GRID_SIZE)
                      for y in np.linspace(0, IMG_DIM-ATT_DIM-1, LOC_GRID_SIZE)]
# birds
BIRD_IMG_DIM = 128
BIRD_ATT_DIM = 32
BIRD_ATT_STRIDE_FRACTION = 4  # i.e. overlap of a quarter
MAX_BIRD_GLIMPSES = 5
BIRDS_N_ATTRIBUTES = 312
BIRDS_N_CLASSES = 200
bird_coord_range = list(range(0, BIRD_IMG_DIM-BIRD_ATT_DIM+1,
                              BIRD_ATT_DIM//BIRD_ATT_STRIDE_FRACTION))
POSSIBLE_BIRD_LOCATIONS = [(r, c)
                           for r in bird_coord_range
                           for c in bird_coord_range]

# image retrieval parameters-------------------
# CelebA-HQ
PCA_COMPONENTS = 256
FIT_PCA_BATCH_SIZE = 1000
N_DATA_POINTS_PCA = 1500000
FAKE_CELEBHQ_N_WORKERS = 11
# Birds
BIRDS_PCA_COMPONENTS = 256
BIRDS_PCA_BATCH_SIZE = 1000
BIRDS_N_DATA_POINTS_PCA = 1500000
FAKE_BIRDS_N_WORKERS = 11


def celebhq_importance_sampling_args(true_image, locations):
    noise_var = [0, 5, 10, 20, 40, 80][len(locations)]
    return {"true_image": true_image,
            "locations": locations,
            "n_p_samples": 200,
            "n_q_samples": 1000,
            "noise_var": noise_var,
            "n_chunks_p": 10,
            "n_chunks_q": 10,
            "max_shift_p": 4,
            "max_shift_q": 2,
            "stride_p": 1,
            "stride_q": 2}


# AVP-CNN parameters --------------------------
# both datasets
MAX_TRAINING_GLIMPSES = MAX_GLIMPSES
DEFAULT_CNN_TRAINING_BATCH_SIZE = 64
DEFAULT_CNN_TRAINING_EPOCHS = 2000
DEFAULT_CNN_TRAINING_N_WORKERS = 7
DEFAULT_CNN_TRAINING_LR = 1e-4
DEFAULT_CNN_ARCHITECTURE = 'densenet'
OED_DEFAULT_BATCH_SIZE = 200

# image generation parameters------------------
# CelebA-HQ
IMAGE_GEN_BATCH_SIZE = 10
# Birds
FINEGAN_BATCH_SIZE = 25

# CelebHQ miscellaneous------------------------
CELEB_ATTRIBUTES = ['5 o Clock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes',
                    'Bald', 'Bangs', 'Big Lips', 'Big Nose', 'Black Hair', 'Blond Hair',
                    'Blurry', 'Brown Hair', 'Bushy Eyebrows', 'Chubby', 'Double Chin',
                    'Eyeglasses', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones',
                    'Male', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard',
                    'Oval Face', 'Pale Skin', 'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks',
                    'Sideburns', 'Smiling', 'Straight Hair', 'Wavy Hair', 'Wearing Earrings',
                    'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace', 'Wearing Necktie',
                    'Young']
CELEBHQ_MEAN = [0.485, 0.456, 0.406]
CELEBHQ_STD = [0.229, 0.224, 0.225]

# Birds miscellaneous--------------------------
BIRDS_MEAN = [0.44] * 3
BIRDS_STD = [0.12] * 3
