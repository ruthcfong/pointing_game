#!/usr/bin/env bash
# Download finetuned network weights.

download_links=("https://www.dropbox.com/s/p0gepxvp8dsybu7/voc_2007_vgg16_checkpoint_best.pth.tar"
                "https://www.dropbox.com/s/i2q9iv6zbc54zpg/coco_2014_vgg16_checkpoint_best.pth.tar"
                "https://www.dropbox.com/s/sezvvozpu1tr65r/voc_2007_inception_v3_checkpoint_best.pth.tar"
                "https://www.dropbox.com/s/47kcjsoeo1os7ck/voc_2007_resnet50_checkpoint_best.pth.tar"
                "https://www.dropbox.com/s/dy888grjj4b1oog/voc_2007_googlenet_checkpoint_best.pth.tar"
                "https://www.dropbox.com/s/i8eskuyn9r0br3d/resnet50_mscoco_Jianming.pth.tar"
                "https://www.dropbox.com/s/qq0kjsorx6gwwpe/resnet50_pascal07_Jianming.pth.tar"
                "https://www.dropbox.com/s/q8rtbizw4caubpd/vgg16_mscoco_Jianming.pth.tar"
                "https://www.dropbox.com/s/7pq3g31sf34l0xy/vgg16_pascal07_Jianming.pth.tar"
                "https://www.dropbox.com/s/jnqvp5tab1thl41/resnet50_pascal07_Jianming_pytorch.pth.tar"
                "https://www.dropbox.com/s/fsx0p4nr7fim96f/resnet50_mscoco_Jianming_pytorch.pth.tar"
                "https://www.dropbox.com/s/idylan4kijox1ug/coco_2014_googlenet_checkpoint_best.pth.tar"
                "https://www.dropbox.com/s/9orapxnc24s0rf1/coco_2014_resnet50_checkpoint_best.pth.tar"
                )

mkdir models
cd models

for download_link in "${download_links[@]}"
do
    wget --continue ${download_link}
done
