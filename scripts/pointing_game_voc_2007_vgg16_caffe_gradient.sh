#!/usr/bin/env bash
# Change these.
dataset="voc_2007"
arch="vgg16"
# checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint_best.pth.tar"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game_rise/model_weights/vgg16_pascal07_Jianming.pth.tar"
gpu=1
final_gap_layer=1
#data_dir="/datasets/MSCOCO"
#ann_dir="/scratch/shared/slow/ruthfong/coco/annotations"
#split="val2014"
converted_caffe=1

# Keep these hyper-parameters.
metric="pointing"
vis_method="gradient"
smooth_sigma="0.02"
tolerance=15
split="test"

out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}_${split}/${arch}_caffe_v2/${vis_method}_${metric}_smooth_${smooth_sigma}_tolerance_${tolerance}.txt"
save_dir="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}_${split}/${arch}_caffe_v2/${vis_method}/smooth_${smooth_sigma}_tolerance_${tolerance}"

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --gpu ${gpu} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --final_gap_layer ${final_gap_layer} \
    --vis_method ${vis_method} \
    --dataset ${dataset} \
    --out_path ${out_path} \
    --metric ${metric} \
    --save_dir ${save_dir} \
    --converted_caffe ${converted_caffe} \
    --split ${split}
    # --start_index ${start_index} \
    # --end_index ${end_index} \
    # --data_dir ${data_dir} \
    # --ann_dir ${ann_dir} \
    # --split ${split} \
