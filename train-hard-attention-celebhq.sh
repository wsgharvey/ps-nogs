if [ "$1" == "" ] || [ "$1" == "RAM" ]; then
    NSEQS="0"
else
    NSEQS="600"
fi
if [ "$2" == "" ]; then
    ATTR="Male"
else
    ATTR=$2
fi
if [ "$3" == "" ]; then
    SEED="1"
else
    SEED=$3
fi

cd training-hard-attention

# train artifact
python main.py --dataset celebhq --attr $ATTR --n_optimal_seqs $NSEQS --seed $SEED --glimpse_net_type standard_fc_False --conv_depth 1-16-1-32-0-32 --hidden_size 64 --epochs 300

# test artifact (and save results)
python main.py --dataset celebhq --attr $ATTR --n_optimal_seqs $NSEQS --seed $SEED --glimpse_net_type standard_fc_False --conv_depth 1-16-1-32-0-32 --hidden_size 64 --epochs 300 --mode test

# print results reported in paper
python main.py --dataset celebhq --attr $ATTR --n_optimal_seqs $NSEQS --seed $SEED --glimpse_net_type standard_fc_False --conv_depth 1-16-1-32-0-32 --hidden_size 64 --epochs 300 --mode summarise

