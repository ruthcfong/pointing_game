#!/usr/bin/env bash
# Change these.
dataset="voc_2007"
arch="vgg16"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint_best.pth.tar"
gpu=0
final_gap_layer=1

# Keep these hyper-parameters.
smooth_sigma="0.02"
tolerance=15

vis_method='gradient'
metric='pointing'
split='test'

out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}/${arch}/${vis_method}_${metric}_smooth_${smooth_sigma}_tolerance_${tolerance}.txt"

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --final_gap_layer ${final_gap_layer} \
    --metric ${metric} \
    --split ${split} \
    --vis_method ${vis_method} \
    --out_path ${out_path} \
    --debug 1 # \
    # --gpu ${gpu}
