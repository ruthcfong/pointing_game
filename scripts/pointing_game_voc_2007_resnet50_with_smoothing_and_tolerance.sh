#!/usr/bin/env bash
# Change these.
arch="resnet50"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/voc_2007_resnet50_checkpoint_best.pth.tar"
gpu=0

# Keep these hyper-parameters.
smooth_sigma="0.02"
tolerance=15

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --gpu ${gpu} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --arch ${arch}
