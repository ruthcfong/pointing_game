#!/usr/bin/env bash
# Change these.
dataset="voc_2007"
arch="vgg16"
vis_method="rise"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint_best.pth.tar"
gpu=3
final_gap_layer=1

# Keep these hyper-parameters.
smooth_sigma="0.02"
tolerance=15

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --dataset ${dataset} \
    --arch ${arch} \
    --gpu ${gpu} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --final_gap_layer ${final_gap_layer} \
    --vis_method ${vis_method}
