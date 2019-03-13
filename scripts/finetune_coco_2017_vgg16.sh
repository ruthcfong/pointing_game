#!/usr/bin/env bash
# Leave these hyper-parameters as is.
arch="vgg16"
dataset="coco_2017"
lr="1e-2"
batch_size=32
train_split="train2017"
val_split="val2017"

# Change the paths.
data_dir="/scratch/shared/slow/ruthfong/coco/images"
ann_dir="/scratch/shared/slow/ruthfong/coco/annotations"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint.pth.tar"

# Set gpu.
gpu=2

python finetune.py --data_dir ${data_dir} \
    --checkpoint_path ${checkpoint_path} \
    --dataset ${dataset} \
    --arch ${arch} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --train_split ${train_split} \
    --val_split ${val_split} \
    --ann_dir ${ann_dir} \
    --gpu ${gpu}
