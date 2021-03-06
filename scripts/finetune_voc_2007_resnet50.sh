#!/usr/bin/env bash
# Leave these hyper-parameters as is.
arch="resnet50"
dataset="voc_2007"
lr="1e-2"
batch_size=64
optimizer="Adam"
num_reduces=4

# Change the paths.
data_dir="/datasets/pascal"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_${optimizer}_r${num_reduces}_checkpoint.pth.tar"
#resume=1

# Set gpu.
gpu=0

python finetune.py --data_dir ${data_dir} \
    --checkpoint_path ${checkpoint_path} \
    --dataset ${dataset} \
    --arch ${arch} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --gpu ${gpu} \
    --optimizer ${optimizer} \
    --num_reduces ${num_reduces} # \
    #--resume ${resume}
