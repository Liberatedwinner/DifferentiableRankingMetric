#!/bin/zsh
echo $1 # device_id
echo $2 # num_pos
for dset in sk-712 epinion-712 ml-1m-712 pinterest-712 melon-712 ml-20m-712
do
    python mp-ours-pt.py --dataset_name=$dset --device_id=$1 --pos_sample=$2 --model_name=k=$2
done
