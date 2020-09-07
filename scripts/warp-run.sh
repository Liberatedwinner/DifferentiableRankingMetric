#!/bin/sh


for dset in sk-712 epinion-712 ml-1m-712 pinterest-712 melon-712 ml-20m-712

do
    python warp-pt.py --dataset_name=$dset --model_name=warp
done
