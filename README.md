Code (and generated near-optimal glimpse sequences) for `Near-Optimal Glimpse
Sequences for Improved Hard Attention Neural Network Training`. Training hard
attention directory based on
`https://github.com/kevinzakka/recurrent-visual-attention`.

We ran experiments in a python 3.8 virtual environment. Install the requirements
with `pip install -r requirements.txt`. CUDA is required for generating the
near-optimal sequences.

Downloading the datasets: we obtained the CelebA-HQ dataset from [this github
repo](https://github.com/nperraud/download-celebA-HQ). We provide the script
`download-celebhq.sh` to more easily download it. To use this, first download
`https://drive.google.com/file/d/1xYnk8eU0zJLoX5iVt_FGjJxDYKj3VsN_/view?usp=sharing`
and save to `data/celebhq/images.zip` before running `bash download-celebhq.sh`.
For CUB, download it from
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
(http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
into `data/birds` and untar. Process using `python preprocess_birds_dataset.py`.

# Generating NOGS
The `optimal-sequences` directory contains the near-optimal glimpse sequences
generated in our experiments, in the form of sequences of pixel coordinates for
glimpses 1 to T. These are the coordinates of the top left pixel in each
glimpse. See `generating-near-optimal-glimpse-sequences/README.md` for how to
generate more, or replicate these. 

# Training hard attention with partial supervision
The directory `training-hard-attention` contains the code for using these (or
our RAM baseline) to train hard attention networks. The script
`train-hard-attention-celebhq.sh` runs our experiments for this. To rerun our
experiments with, e.g., `PS-NOGS` with attribute `Bags_Under_Eyes` and seed 1,
use: 
```
bash train-hard-attention-celebhq.sh PS-NOGS Bags_Under_Eyes 1
```

To run the RAM baseline, use e.g.:
```
bash train-hard-attention-celebhq.sh RAM Bags_Under_Eyes 1
```
