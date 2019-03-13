#!/usr/bin/env bash
# Keep these hyper-parameters.
vis_method="gradient"
smooth_sigma="0.00"
tolerance=0

# Change these.
dataset="voc_2007"
arch="vgg16"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint_best.pth.tar"
gpu=1
final_gap_layer=1

metric="average_precision"
threshold_type="mean"
alpha="0.5"

out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}_${arch}_${vis_method}_${metric}_${threshold_type}_${alpha}_smooth_${smooth_sigma}_tolerance_${tolerance}.txt"

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --gpu ${gpu} \
    --vis_method ${vis_method} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --final_gap_layer ${final_gap_layer} \
    --metric ${metric} \
    --alpha ${alpha} \
    --out_path ${out_path}

