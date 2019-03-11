#!/usr/bin/env bash
# Change these.
dataset="coco_2014"
arch="vgg16"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint_best.pth.tar"
gpu=2
final_gap_layer=0
data_dir="/datasets/MSCOCO"
ann_dir="/scratch/shared/slow/ruthfong/coco/annotations"
split="val2014"

# Keep these hyper-parameters.
vis_method="gradient"
smooth_sigma="0.02"
tolerance=15

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --gpu ${gpu} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --final_gap_layer ${final_gap_layer} \
    --data_dir ${data_dir} \
    --ann_dir ${ann_dir} \
    --split ${split} \
    --vis_method ${vis_method} \
    --dataset ${dataset}

