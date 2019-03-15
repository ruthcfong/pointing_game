#!/usr/bin/env bash
# Change these.
dataset="voc_2007"
arch="vgg16"
vis_method="rise"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint_best.pth.tar"
gpu=0
final_gap_layer=1

# Keep these hyper-parameters.
smooth_sigma="0.00"
tolerance=15
metric="pointing"

start_index=4000
end_index=4952

out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}/${arch}/${vis_method}_${metric}_smooth_${smooth_sigma}_tolerance_${tolerance}_${start_index}_${end_index}.txt"
save_dir="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}/${arch}/${vis_method}/smooth_${smooth_sigma}_tolerance_${tolerance}"

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --dataset ${dataset} \
    --arch ${arch} \
    --gpu ${gpu} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --final_gap_layer ${final_gap_layer} \
    --vis_method ${vis_method} \
    --save_dir ${save_dir} \
    --start_index ${start_index} \
    --end_index ${end_index}
