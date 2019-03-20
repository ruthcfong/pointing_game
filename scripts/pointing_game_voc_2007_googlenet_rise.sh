#!/usr/bin/env bash
# Change these.
dataset="voc_2007"
arch="googlenet"
vis_method="rise"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint_best.pth.tar"
gpu=1
final_gap_layer=1

# Keep these hyper-parameters.
smooth_sigma="0.02"
tolerance=15
metric="pointing"

start_index=0
#end_index=2500
end_index=4952

out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}/${arch}/${vis_method}_${metric}_smooth_${smooth_sigma}_tolerance_${tolerance}.txt"
save_dir="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}/${arch}/${vis_method}/smooth_0.00_tolerance_15"

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --dataset ${dataset} \
    --arch ${arch} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --final_gap_layer ${final_gap_layer} \
    --vis_method ${vis_method} \
    --save_dir ${save_dir} \
    --gpu ${gpu} \
    --metric ${metric} \
    --out_path ${out_path} \
    --start_index ${start_index} \
    --end_index ${end_index}
    # --load_from_save_dir 1
