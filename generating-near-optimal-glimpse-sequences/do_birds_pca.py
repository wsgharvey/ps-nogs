import os
import argparse
from mea.image_sampling.birds.pca import fit_pca, make_latents
from mea.config import FAKE_BIRDS_PCA_PATH
from mea.utils import set_random_seed

set_random_seed(0)  # I think this should be deterministic anyway but best to be safe

parser = argparse.ArgumentParser(description='Fit PCA on birds dataset.')
parser.add_argument('--image_type', type=str, default='images',
                    help='Type of image to perform PCA on.')
parser.add_argument('--N0', type=int, default=0,
                    help='First index.')
parser.add_argument('--N', type=int, default=-1,
                    help='Number of images to fit on.')
args = parser.parse_args()

if not os.path.exists(FAKE_BIRDS_PCA_PATH):
    os.makedirs(FAKE_BIRDS_PCA_PATH)

fit_pca(args.image_type,
        args.N)

make_latents(args.image_type,
             args.N0, args.N)
