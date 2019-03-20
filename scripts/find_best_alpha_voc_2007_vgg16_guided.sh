#!/usr/bin/env bash
# Keep these hyper-parameters.
vis_method="guided_backprop"
smooth_sigma="0.02"
tolerance=0

# Change these.
dataset="voc_2007"
arch="vgg16"
split="val"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint_best.pth.tar"
gpu=1
final_gap_layer=1

metric="average_precision"
threshold_type="energy"

out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}/${arch}/${metric}/${vis_method}_${split}/${threshold_type}_smooth_${smooth_sigma}_tolerance_${tolerance}"

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --gpu ${gpu} \
    --vis_method ${vis_method} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --final_gap_layer ${final_gap_layer} \
    --metric ${metric} \
    --out_path ${out_path} \
    --find_best_alpha
