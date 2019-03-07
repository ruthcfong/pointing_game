#!/usr/bin/env bash
# Leave these hyper-parameters as is.
arch="vgg16"
dataset="voc_2007"
lr="1e-2"
batch_size=64

# Change the paths.
data_dir="/datasets/pascal"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint.pth.tar"

# Set gpu.
gpu=1

python finetune.py --data_dir ${data_dir} \
    --checkpoint_path ${checkpoint_path} \
    --dataset ${dataset} \
    --arch ${arch} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --gpu ${gpu}