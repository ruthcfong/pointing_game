#!/usr/bin/env bash
# Change these.
dataset="voc_2007"
arch="vgg16"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint_best.pth.tar"
vis_method="guided_backprop"
gpu=0

# Keep these hyper-parameters.
smooth_sigma="0.02"
tolerance=15

out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}/${arch}/${vis_method}_${metric}_smooth_${smooth_sigma}_tolerance_${tolerance}.txt"

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --vis_method ${vis_method} \
    --arch ${arch} \
    --gpu ${gpu} \
    --out_path ${out_path}
