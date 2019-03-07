# pointing_game

This repo implements the pointing game originally introduced in Zhang et al., ECCV 2016. "Top-down Neural Attention by Excitation Backprop." in PyTorch (the original paper's [code](https://github.com/jimmie33/Caffe-ExcitationBP/) usees caffe/pycaffe).

## TODO
* Support COCO dataset (currently only VOC 2007 is supported).
* Support other visualizations besides gradient.
* Add more pre-trained, fine-tuned networks.
* Add more results.

## 1a. Download pre-trained, fine-tuned networks.
Run `bash scripts/download_finetuned_networks.sh` to download my finetuned networks.

### [VGG16 pre-trained on ImageNet and fine-tuned on VOC 2007](https://www.dropbox.com/s/p0gepxvp8dsybu7/voc_2007_vgg16_checkpoint_best.pth.tar?dl=0)
* Trained using [finetune_voc_2007_vgg16.sh](scripts/finetune_voc_2007_vgg16.sh).
* Batch size = 32, SGD with learning rate 0.01 for 310 epochs (when training plateaued).
* Precision: 0.8604, Recall: 0.7549, Loss: 0.0808

### [ResNet50 pre-trained on ImageNet and fine-tuned on VOC 2007](TODO)
* Trained using [finetune_voc_2007_resnet50.sh](scripts/finetune_voc_2007_resnet50.sh).
* Batch size = 64, SGD with learning rate 0.01 for X epochs (when training plateaued).
* Precision: TBD, Recall: TBD, Loss: TBD.

## 1b. Alternatively, fine-tune your own networks.
Use [finetune.py](finetune.py) to fine-tune your own networks.

## 2. Evaluate performance on pointing game.
Use [pointing_game.py](pointing_game.py) to evaluate a visualization method on the pointing game.

## Results
### VOC2007, VGG16
* Gradient: 0.6929 (no tolerance or gaussian smoothing)
* Gradient: 0.7168 (no tolerance; smooth_sigma = 0.02)
* Gradient: TBD (no smoothing, tolerance = 15)
* Gradient: 0.7626 (tolerance = 15, smooth_sigma = 0.02, as in Zhang et al., ECCV 2016)
