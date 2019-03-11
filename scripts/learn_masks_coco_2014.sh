#!/usr/bin/env bash
data_dir="TODO" # path to root directory for COCO.
ann_dir="TODO"  # path to annotation directory for COCO.
checkpoint_path="TODO" # path to VGG16 weights finetuned for COCO.
dataset="coco_2014"
arch="vgg16"
split="val2014"
input_size=224

python learn_masks_for_pointing.py \
    --data_dir ${data_dir} \
    --checkpoint_path ${checkpoint_path} \
    --ann_dir ${ann_dir} \
    --split ${split} \
    --input_size ${input_size}
