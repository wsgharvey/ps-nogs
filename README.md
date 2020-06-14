Code (and generated near-optimal glimpse sequences) for `Near-Optimal Glimpse Sequences for Improved Hard Attention Neural Network Training`. Training hard attention directory based on `https://github.com/kevinzakka/recurrent-visual-attention`.

We ran experiments in a python 3.8 virtual environment. Install the requirements with `pip install -r requirements.txt`. CUDA is required for generating the near-optimal sequences.

Downloading the datasets: we obtained the CelebA-HQ dataset from [this github repo](https://github.com/nperraud/download-celebA-HQ). The images should be saved in `data/celebhq/images`. Their names should match those in `data/celebhq/annotations.txt`. For CUB, download it from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html (http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) into `data/birds` and untar. Process using `python preprocess_birds_dataset.py`.

The `optimal-sequences` directory contains the near-optimal glimpse sequences generated in our experiments, in the form of sequences of pixel coordinates for glimpses 1 to T. These are the coordinates of the top left pixel in each glimpse. See `generating-near-optimal-glimpse-sequences/README.md` for how to generate more, or replocate these. See `training-hard-attention/README.md` for how to using these (or the RAM baseline) to train hard attention networks.
