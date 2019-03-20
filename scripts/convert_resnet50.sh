#!/usr/bin/env bash
#dset_name="pascal07"
dset_name="mscoco"
orig_path="/scratch/shared/slow/ruthfong/pointing_game_rise/model_weights/resnet50_${dset_name}_Jianming.pth.tar"
new_path="/scratch/shared/slow/ruthfong/pointing_game_rise/model_weights/resnet50_${dset_name}_Jianming_pytorch.pth.tar"

python2 convert_resnet50_to_pytorch.py ${orig_path} ${new_path}
