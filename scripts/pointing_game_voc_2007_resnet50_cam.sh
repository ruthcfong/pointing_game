#!/usr/bin/env bash
# Change these.
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/voc_2007_resnet50_checkpoint_best.pth.tar"
arch="resnet50"
vis_method="cam"
gpu=1
debug=1

# Keep these hyper-parameters.
smooth_sigma="0.02"
tolerance=15

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --vis_method ${vis_method} \
    --arch ${arch} \
    --debug ${debug} \
    --gpu ${gpu}
