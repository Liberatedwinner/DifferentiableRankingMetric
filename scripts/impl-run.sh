#!/bin/sh

echo $1 # model_name

for dset in sk-712 epinion-712 ml-1m-712 pinterest-712 melon-712 ml-20m-712

do
    python implicit-pt.py --dataset_name=$dset --model_name=$1
done
