import os
import pickle
import torch
import time

from mea.image_sampling import ImportanceSampler, ESS
from mea.variational_cnn.dropout_img import batch_img_dropout_with_channel
from mea.dataset import CelebHQ, to_normalized_tensor
from mea.config import CELEB_ATTRIBUTES, OED_INDICES, OED_DEFAULT_BATCH_SIZE, \
    POSSIBLE_LOCATIONS, celebhq_importance_sampling_args, GET_OED_LOG_ATTR_DIR, \
    GET_OPTIMAL_SEQUENCE_PATH, GET_ENTROPY_MAP_PATH, GET_OPTIMAL_SEQUENCE_ATTR_DIR, \
    MAX_GLIMPSES, CNN_SAVE_PATH, GET_IMAGE_SAMPLE_LOG_PATH, GET_OED_FIRST_LOC_PATH, \
    ATT_DIM
from mea.utils import bernoulli_entropy, set_random_seed, read_sequence, write_sequence
import argparse


parser = argparse.ArgumentParser(description='Attention CNN')
parser.add_argument('index', type=int,
                    help='Index of first image to annotate.')
parser.add_argument('--n', type=int, default=1,
                    help='Number of images to annotate.')
parser.add_argument('--attr', type=int,
                    help='Class label to use.')
parser.add_argument('--trained-cnn', type=str,
                    help='Path of trained variational CNN.')
parser.add_argument('--batch-size', type=int, default=OED_DEFAULT_BATCH_SIZE,
                    help='Size of batches to input to CNN.')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed.')
args = parser.parse_args()
CUDA = True
CLASS_INDEX = args.attr
INIT_IMG = args.index
N_IMGS = args.n
BATCH_SIZE = args.batch_size
SEED = args.seed

print(f"CLASS {CLASS_INDEX} ({CELEB_ATTRIBUTES[CLASS_INDEX]})")
print(f"IMAGES {INIT_IMG} TO {INIT_IMG+N_IMGS-1}.")


# load dataset and image sampler-----------------------------------------------------------------------
sampler = ImportanceSampler(cuda=CUDA)
CNN_PATH = args.trained_cnn
q = pickle.load(open(CNN_PATH, 'rb'))
if CUDA:
    q = q.cuda()
dataset = CelebHQ(to_normalized_tensor, OED_INDICES)

# check save paths exist-------------------------------------------------------------------------------
seq_dir = GET_OPTIMAL_SEQUENCE_ATTR_DIR(CLASS_INDEX)
if not os.path.isdir(seq_dir):
    os.makedirs(seq_dir)
epe_dir = GET_OED_LOG_ATTR_DIR(CLASS_INDEX)
if not os.path.isdir(epe_dir):
    os.makedirs(epe_dir)

# check if we have cached the first location-----------------------------------------------------------
CNN_MOD_TIME = os.path.getmtime(CNN_PATH)
first_loc_path = GET_OED_FIRST_LOC_PATH(
        seed=SEED,
        attr=CLASS_INDEX,
        net_path=args.trained_cnn,
        mod_time=CNN_MOD_TIME
        )
if os.path.exists(first_loc_path):
    first_loc = read_sequence(first_loc_path)[0]
else:
    first_loc = None

# carry out the experimental design for each image-----------------------------------------------------
for img_no in range(INIT_IMG, INIT_IMG+N_IMGS):
    set_random_seed(args.seed)

    true_image, _ = dataset[img_no]
    true_image = true_image.squeeze(0)
    if CUDA:
        true_image = true_image.cuda()

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

    for t in range(INIT_T, MAX_GLIMPSES):
        print(f"Time {t}")

        start = time.time()
        sampled_images, weights, indices, flips = sampler.posterior_samples(
            **celebhq_importance_sampling_args(true_image, observed_locations),
            return_indices=True
        )
        image_samples.append((weights.cpu(), indices, flips))
        print(f"Sampled images in {time.time()-start}.")
        # batch locations according to how many images we have sampled
        locs_per_batch = BATCH_SIZE // len(weights)
        def possible_location_batches():
            i = 0
            while i < len(POSSIBLE_LOCATIONS):
                yield POSSIBLE_LOCATIONS[i:min(i+locs_per_batch, len(POSSIBLE_LOCATIONS))]
                i += locs_per_batch
        EPEs.append({})
        # batch as many locations together as fit on GPU (for performance)
        for location_batch in possible_location_batches():
            with torch.no_grad():
                batch = []
                for location in location_batch:
                    batch.append(batch_img_dropout_with_channel(
                        sampled_images,
                        (ATT_DIM, ATT_DIM),
                        locations=observed_locations+[location],
                    ))
                batch = torch.cat(batch, dim=0)
                # get posteriors
                posteriors = q(batch)[:, CLASS_INDEX]
                # get posterior entropies
                entropies = bernoulli_entropy(posteriors)
                # get expected entropy for each location in batch
                entropies = entropies.view(len(location_batch), len(weights))
                expected_entropies = torch.mv(entropies, weights).cpu().numpy()
                for location, EPE in zip(location_batch, expected_entropies):
                    EPEs[-1][location] = EPE

        # take best location
        best_location = min(EPEs[-1], key=lambda x: EPEs[-1][x])
        observed_locations.append(best_location)
        print(f"Calculated expected entropies in {time.time()-start}s.")
        print(f"Looking at {best_location}.")
        print(f"Calculated with ESS of {ESS(weights.cpu())}")

    write_sequence(GET_OPTIMAL_SEQUENCE_PATH(CLASS_INDEX, img_no),
                   observed_locations)

    pickle.dump(EPEs, open(GET_ENTROPY_MAP_PATH(CLASS_INDEX, img_no), 'wb'))
    pickle.dump(image_samples, open(GET_IMAGE_SAMPLE_LOG_PATH(CLASS_INDEX, img_no), 'wb'))

    if first_loc is None:
        first_loc = observed_locations[0]
        write_sequence(first_loc_path, observed_locations[:1])
