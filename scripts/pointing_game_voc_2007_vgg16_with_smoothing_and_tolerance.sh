#!/usr/bin/env bash
# Change these.
checkpoint_path="models/voc_2007_vgg16_checkpoint_best.pth.tar"
gpu=2

# Keep these hyper-parameters.
smooth_sigma="0.02"
tolerance=15

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --gpu ${gpu} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance}
