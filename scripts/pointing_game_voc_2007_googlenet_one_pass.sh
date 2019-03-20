#!/usr/bin/env bash
# Change these.
dataset="voc_2007"
arch="googlenet"
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_checkpoint_best.pth.tar"
optimizer="Adam"
num_reduces=4
checkpoint_path="/scratch/shared/slow/ruthfong/pointing_game/${dataset}_${arch}_${optimizer}_r${num_reduces}_checkpoint_best.pth.tar"
gpu=1
debug=0

# Keep these hyper-parameters.
smooth_sigma="0.02"
tolerance=15
metrics=("pointing")
vis_methods=("gradient" "guided_backprop" "cam" "grad_cam")
#vis_methods=( "grad_cam" )

for metric in "${metrics[@]}"
do
    for vis_method in "${vis_methods[@]}"
    do
        out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}/${arch}_${optimizer}_r${num_reduces}_v2/${vis_method}_${metric}_smooth_${smooth_sigma}_tolerance_${tolerance}.txt"
        #out_path="/scratch/shared/slow/ruthfong/pointing_game/results/${dataset}/${arch}/${vis_method}_${metric}_smooth_${smooth_sigma}_tolerance_${tolerance}.txt"
        python pointing_game.py --checkpoint_path ${checkpoint_path} \
            --smooth_sigma ${smooth_sigma} \
            --tolerance ${tolerance} \
            --vis_method ${vis_method} \
            --arch ${arch} \
            --debug ${debug} \
            --gpu ${gpu} \
            --out_path ${out_path} \
            --metric ${metric} \
            --vis_method ${vis_method}
    done
done
