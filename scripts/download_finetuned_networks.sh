#!/usr/bin/env bash
# Download finetuned network weights.

download_links=("https://www.dropbox.com/s/p0gepxvp8dsybu7/voc_2007_vgg16_checkpoint_best.pth.tar"
                "https://www.dropbox.com/s/i2q9iv6zbc54zpg/coco_2014_vgg16_checkpoint_best.pth.tar"
                "https://www.dropbox.com/s/sezvvozpu1tr65r/voc_2007_inception_v3_checkpoint_best.pth.tar"
                "https://www.dropbox.com/s/47kcjsoeo1os7ck/voc_2007_resnet50_checkpoint_best.pth.tar"
                "https://www.dropbox.com/s/dy888grjj4b1oog/voc_2007_googlenet_checkpoint_best.pth.tar" )

mkdir models
cd models

for download_link in "${download_links[@]}"
do
    wget ${download_link}
done
