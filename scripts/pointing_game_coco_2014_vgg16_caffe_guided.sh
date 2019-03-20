#!/usr/bin/env bash
# Change these.
dataset="coco_2014"
arch="vgg16"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game_rise/model_weights/vgg16_mscoco_Jianming.pth.tar"
gpu=1
final_gap_layer=1
data_dir="/datasets/MSCOCO"
ann_dir="/scratch/shared/slow/ruthfong/coco/annotations"
split="val2014"
converted_caffe=1

# Keep these hyper-parameters.
metric="pointing"
vis_method="guided_backprop"
smooth_sigma="0.02"
tolerance=15

start_index=0
end_index=10000
#end_index=40137

out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}_${split}/${arch}_caffe/${vis_method}_${metric}_smooth_${smooth_sigma}_tolerance_${tolerance}_${start_index}_${end_index}.txt"
save_dir="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}_${split}/${arch}_caffe/${vis_method}/smooth_${smooth_sigma}_tolerance_${tolerance}"

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
    --start_index ${start_index} \
    --end_index ${end_index} \
    --metric ${metric} \
    --save_dir ${save_dir} \
    --converted_caffe ${converted_caffe}
