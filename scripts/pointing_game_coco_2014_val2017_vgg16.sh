#!/usr/bin/env bash
# Change these.
dataset="coco_2014"
arch="vgg16"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint_best.pth.tar"
gpu=1
final_gap_layer=1
#data_dir="/datasets/MSCOCO"
data_dir="/scratch/shared/slow/ruthfong/coco/images"
ann_dir="/scratch/shared/slow/ruthfong/coco/annotations"
split="val2017"

# Keep these hyper-parameters.
metric="pointing"
vis_method="gradient"
smooth_sigma="0.02"
tolerance=15

#start_index=0
#end_index=10000
#end_index=40137

out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}_${split}/${arch}/${vis_method}_${metric}_smooth_${smooth_sigma}_tolerance_${tolerance}_${start_index}_${end_index}.txt"
save_dir="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}_${split}/${arch}/${vis_method}/smooth_0.00_tolerance_15"

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --gpu ${gpu} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --final_gap_layer ${final_gap_layer} \
    --data_dir ${data_dir} \
    --ann_dir ${ann_dir} \
    --split ${split} \
    --vis_method ${vis_method} \
    --dataset ${dataset} \
    --out_path ${out_path} \
    --metric ${metric} \
    --save_dir ${save_dir}
