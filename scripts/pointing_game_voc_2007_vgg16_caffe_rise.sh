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
vis_method="rise"
smooth_sigma="0.02"
tolerance=15
split="test"
debug=1
--gpu_batch 500

# start_index=0
# end_index=2500
start_index=2500
# end_index=2500

out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}_${split}/${arch}_caffe_v3/${vis_method}_${metric}_smooth_${smooth_sigma}_tolerance_${tolerance}.txt"
save_dir="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}_${split}/${arch}_caffe_v3/${vis_method}/smooth_${smooth_sigma}_tolerance_${tolerance}"

python pointing_game.py --checkpoint_path ${checkpoint_path} \
    --smooth_sigma ${smooth_sigma} \
    --tolerance ${tolerance} \
    --final_gap_layer ${final_gap_layer} \
    --vis_method ${vis_method} \
    --dataset ${dataset} \
    --out_path ${out_path} \
    --metric ${metric} \
    --save_dir ${save_dir} \
    --converted_caffe ${converted_caffe} \
    --split ${split} \
    --gpu ${gpu} \
    --start_index ${start_index} \
    --gpu_batch ${gpu_batch}
    # --end_index ${end_index}
    # --debug ${debug} \
