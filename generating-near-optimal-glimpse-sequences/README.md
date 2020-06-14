# Generating near-optimal sequences for hard-attention training
This repository contains the code for generating near-optimal sequences of
glimpse locations with Bayesian experimental design. The code for using these to
supervise the training of hard attention mechanisms, and for generating
heuristic supervision sequences, has not yet been released.

## Steps to generate near-optimal sequences:

1. Create a dataset for the image retrieval. First download pre-trained GAN
   weights. Use ``bash download-stylegan-weights.sh`` for CelebA-HQ, or for CUB
   download the weights for FineGAN (released with the FineGAN paper) and save
   to `finegan-weights/birds.pt`. Then, the fake CelebA-HQ dataset we use can be
   generated in full with e.g. ``python generate_dataset.py --N0 0 --N
   1500000``, or in part by specifying different `N0` (the index of the initial
   image to generate) and `N` (the number of images to generate). The fake CUB
   dataset can be generated similarly using `generate_dataset.py`.

2. An AVP-CNN can be trained for CelebA-HQ with `python train_variational_cnn.py
   test` or CUB with `python train_bird_variational_cnn.py test`.

3. After the dataset has been generated, we perform principal component analysis
   on it, using ``python do_pca.py`` for CelebA-HQ or `python do_birds_pca.py`
   for CUB.
   
4. We can now begin generating optimal sequences. This is done for CelebA-HQ for
   image index IMG and attribute index ATTR (0...39, in alphabetical order)
   using ``python oed.py --index IMG --class-index ATTR --trained-cnn PATH``
   where PATH is the path to one of the AVP-CNN checkpoints (e.g.
   `trained-nets/test_seed_0_epoch_0_BEST.p`). Similarly, it can be done for CUB
   with `python bird_oed.py IMG --trained-cnn PATH` (where PATH is e.g.
   `trained-nets/birds/test_seed_0_epoch_0_BEST.p`).
