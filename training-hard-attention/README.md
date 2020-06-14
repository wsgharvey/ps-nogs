Code for `Near-Optimal Glimpse Sequences for Improved Hard Attention Neural Network Training`. This directory is based on `https://github.com/kevinzakka/recurrent-visual-attention`.

This assumes you have the datasets configured as described in `../README.md`.

Train PS-NOGS (for CelebA-HQ attribute Narrow_Eyes) with, e.g.:
```
python main.py --dataset celebhq --attr Male --n_optimal_seqs 600 --seed 1 --glimpse_net_type standard_fc_False --conv_depth 1-16-1-32-0-32 --hidden_size 64
```

Train RAM with:
```
python main.py --dataset celebhq --attr Male --n_optimal_seqs 0 --seed 1 --glimpse_net_type standard_fc_False --conv_depth 1-16-1-32-0-32 --hidden_size 64
```
