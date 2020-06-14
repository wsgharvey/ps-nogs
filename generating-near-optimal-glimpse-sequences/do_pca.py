from mea.image_sampling.pca import fit_pca, make_latents
from mea.utils import set_random_seed

set_random_seed(0)  # I think this should be deterministic anyway but best to be safe

fit_pca()
make_latents()
