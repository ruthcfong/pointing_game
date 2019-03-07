#!/usr/bin/env bash
# Download finetuned network weights.

download_links=("https://www.dropbox.com/s/p0gepxvp8dsybu7/voc_2007_vgg16_checkpoint_best.pth.tar")

mkdir models
cd models

for download_link in "${download_links[@]}"
do
    wget ${download_link}
done
