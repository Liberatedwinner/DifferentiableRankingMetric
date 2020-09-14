#!/bin/bash
echo $1 # device_id
echo $2 # model_name
for dset in epinion-712 ml-20m-712 melon-712
do
    python autoencoder-pt.py --dataset_name=$dset --device_id=$1 --model_name=$2
done

