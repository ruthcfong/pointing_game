#!/usr/bin/env bash
# Change these.
arch="vgg16"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/voc_2007_${arch}_checkpoint_best.pth.tar"
gpu=2
final_gap_layer=1

# Keep these hyper-parameters.
smooth_sigma="0.02"
tolerance=15

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --gpu ${gpu} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --final_gap_layer ${final_gap_layer}
