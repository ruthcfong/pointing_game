#!/usr/bin/env bash
# Leave these hyper-parameters as is.
arch="googlenet"
dataset="voc_2007"
lr="1e-2"
num_reduces=4
batch_size=64
input_size=224
optimizer="Adam"

# Change the paths.
data_dir="/datasets/pascal"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_${optimizer}_r${num_reduces}_checkpoint.pth.tar"

# Set gpu.
gpu=0

python finetune.py --data_dir ${data_dir} \
    --checkpoint_path ${checkpoint_path} \
    --dataset ${dataset} \
    --arch ${arch} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --input_size ${input_size} \
    --gpu ${gpu} \
    --optimizer ${optimizer} \
    --num_reduces ${num_reduces}
