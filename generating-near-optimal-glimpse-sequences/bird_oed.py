import os
from os.path import isdir, join
import pickle
import torch
import time

from mea.image_sampling import BirdImportanceSampler, ESS
from mea.variational_cnn.dropout_img import \
    batch_img_dropout_with_channel
from mea.bird_dataset import Birds, to_normalized_tensor
from mea.config import OED_DEFAULT_BATCH_SIZE, \
    BIRDS_CNN_SAVE_PATH, BIRDS_OPTIMAL_SEQUENCE_DIR, \
    BIRDS_OED_LOG_DIR, MAX_BIRD_GLIMPSES, \
    BIRD_ATT_DIM, GET_BIRDS_OPTIMAL_SEQUENCE_PATH, \
    GET_BIRDS_OED_FIRST_LOC_PATH, \
    GET_BIRDS_ENTROPY_MAP_PATH, \
    GET_BIRDS_IMAGE_SAMPLE_LOG_PATH, \
    POSSIBLE_BIRD_LOCATIONS
from mea.utils import set_random_seed, read_sequence, \
    write_sequence, categorical_entropy
import argparse


parser = argparse.ArgumentParser(description='Attention CNN')
parser.add_argument('index', type=int,
                    default=0,
                    help='Index of first image to annotate.')
parser.add_argument('--n', type=int,
                    default=1,
                    help='Number of images to annotate.')
parser.add_argument('--trained-cnn', type=str,
                    help='Path of trained variational CNN.')
parser.add_argument('--batch-size', type=int,
                    default=OED_DEFAULT_BATCH_SIZE,
                    help='Size of batches to input to CNN.')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed.')
args = parser.parse_args()
CUDA = True
DEVICE = 'cuda' if CUDA else 'cpu'
INIT_IMG = args.index
N_IMGS = args.n
BATCH_SIZE = args.batch_size
SEED = args.seed

print(f"IMAGES {INIT_IMG} TO {INIT_IMG+N_IMGS-1}.")


# load dataset and image sampler--------------------------------------
sampler = BirdImportanceSampler(
    cuda=CUDA,
    n_chunks=20,
    noise_var_p=80,
    n_samples_p=100,
    n_samples_q=500,
)
CNN_PATH = args.trained_cnn
q = pickle.load(open(CNN_PATH, 'rb'))
if CUDA:
    q = q.cuda()
dataset = Birds(
    to_normalized_tensor,
    mode='valid',
    attribute='class'
)

# ensure save paths exist---------------------------------------------
required_dirs = [BIRDS_OPTIMAL_SEQUENCE_DIR,
                 BIRDS_OED_LOG_DIR]
for d in required_dirs:
    if not isdir(d):
        os.makedirs(d)

# check if we have cached the first location-------------------------
cnn_mod_time = os.path.getmtime(CNN_PATH)
first_loc_path = GET_BIRDS_OED_FIRST_LOC_PATH(
        seed=SEED,
        net_path=args.trained_cnn,
        mod_time=cnn_mod_time
        )
if os.path.exists(first_loc_path):
    first_loc = read_sequence(first_loc_path)[0]
else:
    assert INIT_IMG == 0, "For reproducibility, \
    create cache while doing image 0."
    first_loc = None

# carry out the experimental design for each image--------------------
for img_no in range(INIT_IMG, INIT_IMG+N_IMGS):
    set_random_seed(SEED)

    true_image = dataset[img_no][0]\
        .squeeze(0)\
        .to(DEVICE)

    if first_loc is None:
        INIT_T = 0
        observed_locations = []
        image_samples = []
        EPEs = []
    else:
        INIT_T = 1
        observed_locations = [first_loc]
        image_samples = [None]
        EPEs = [None]
        print("Using first location from", first_loc_path)

    for t in range(INIT_T, MAX_BIRD_GLIMPSES):
        print(f"Time {t}")

        start = time.time()
        sampled_images, weights, indices, flips = sampler.sample(
                true_image, observed_locations,
                return_indices=True
                )
        image_samples.append((weights.cpu(), indices, flips))
        print(f"Sampled images in {time.time()-start}.")

        # batch locations according to how many images we have sampled
        locs_per_batch = BATCH_SIZE // len(weights)
        def possible_location_batches():
            i = 0
            L = len(POSSIBLE_BIRD_LOCATIONS)
            while i < L:
                yield POSSIBLE_BIRD_LOCATIONS[i:
                                              min(i+locs_per_batch, L)]
                i += locs_per_batch

        EPEs.append({})
        for location_batch in possible_location_batches():
            with torch.no_grad():
                batch = []
                for location in location_batch:
                    batch.append(batch_img_dropout_with_channel(
                        sampled_images,
                        (BIRD_ATT_DIM, BIRD_ATT_DIM),
                        locations=observed_locations+[location],
                    ))
                batch = torch.cat(batch, dim=0)
                posteriors = q(batch)['softmax'].exp()
                entropies = categorical_entropy(posteriors)\
                    .view(len(location_batch),  # n locations in batch
                          len(weights))         # n images samples
                expected_entropies = torch.mv(
                    entropies,
                    weights
                ).cpu().numpy()
                for location, EPE in zip(location_batch, expected_entropies):
                    EPEs[-1][location] = EPE

        best_location = min(EPEs[-1], key=lambda x: EPEs[-1][x])
        observed_locations.append(best_location)
        print(f"Calculated expected entropies in {time.time()-start}s.")
        print(f"Looking at {best_location}.")
        print(f"Calculated with ESS of {ESS(weights.cpu())}")

    seq_path = GET_BIRDS_OPTIMAL_SEQUENCE_PATH(img_no)
    write_sequence(
        seq_path,
        observed_locations
    )
    image_path = dataset.image_paths[img_no]
    open(seq_path, 'a').write(image_path)

    pickle.dump(
        EPEs,
        open(GET_BIRDS_ENTROPY_MAP_PATH(img_no), 'wb')
    )
    pickle.dump(
        image_samples,
        open(GET_BIRDS_IMAGE_SAMPLE_LOG_PATH(img_no), 'wb')
    )

    if first_loc is None:
        first_loc = observed_locations[0]
        write_sequence(first_loc_path, observed_locations[:1])
