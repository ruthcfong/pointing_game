#!/usr/bin/env bash
# Leave these hyper-parameters as is.
arch="googlenet"
dataset="coco_2014"
lr="1e-2"
batch_size=64
input_size=224
train_split="train2014"
val_split="val2014"

# Change the paths.
data_dir="/datasets/MSCOCO"
ann_dir="/scratch/shared/slow/ruthfong/coco/annotations"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint.pth.tar"

# Set gpu.
gpu=1

python finetune.py --data_dir ${data_dir} \
    --checkpoint_path ${checkpoint_path} \
    --dataset ${dataset} \
    --arch ${arch} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --input_size ${input_size} \
    --gpu ${gpu} \
    --ann_dir ${ann_dir} \
    --train_split ${train_split} \
    --val_split ${val_split}
