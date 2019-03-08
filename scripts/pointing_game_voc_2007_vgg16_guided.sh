#!/usr/bin/env bash
# Change these.
arch="resnet50"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/voc_2007_${arch}_checkpoint_best.pth.tar"
vis_method="guided_backprop"
gpu=2

# Keep these hyper-parameters.
smooth_sigma="0.02"
tolerance=15

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --vis_method ${vis_method} \
    --arch ${arch} \
    --gpu ${gpu}
