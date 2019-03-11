"""
learn_masks_for_pointing.py

Learn masks for the pointing game.
"""
import os

import numpy as np

import torch
from torchvision import datasets, transforms

from utils import (set_gpu, get_device, get_finetune_model,
                   FromVOCToOneHotEncoding, FromCocoToOneHotEncoding,
                   SimpleToTensor)


def learn_masks_for_pointing(data_dir,
                             checkpoint_path,
                             arch='vgg16',
                             dataset='voc_2007',
                             ann_dir=None,
                             split='test',
                             input_size=224):
    """
    Learn explanatory masks for the pointing game.

    Args:
        data_dir: String, path to root directory for dataset.
        checkpoint_path: String, path to checkpoint.
        arch: String, name of torchvision.models architecture.
        dataset: String, name of dataset.
        ann_dir: String, path to root directory containing annotation files
            (used for COCO).
        split: String, name of split.
        input_size: Integer, length of the side of the input image.
    """
    # Load fine-tuned model and convert it to be fully convolutional.
    model = get_finetune_model(arch=arch,
                               dataset=dataset,
                               checkpoint_path=checkpoint_path,
                               convert_to_fully_convolutional=True)
    device = get_device()
    model = model.to(device)

    # Prepare data augmentation.
    assert(isinstance(input_size, int))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Prepare data loaders.
    if 'voc' in dataset:
        year = dataset.split('_')[-1]

        target_transform = transforms.Compose([
            FromVOCToOneHotEncoding(),
            SimpleToTensor(),
        ])

        dset = datasets.VOCDetection(data_dir,
                                     year=year,
                                     image_set=split,
                                     transform=transform,
                                     target_transform=target_transform)
    elif 'coco' in dataset:
        ann_path = os.path.join(ann_dir, 'instances_%s.json' % split)
        target_transform = transforms.Compose([
            FromCocoToOneHotEncoding(),
            SimpleToTensor(),
        ])
        dset = datasets.CocoDetection(os.path.join(data_dir, split),
                                      ann_path,
                                      transform=transform,
                                      target_transform=target_transform)
    else:
        assert(False)

    loader = torch.utils.data.DataLoader(dset,
                                         batch_size=1,
                                         num_workers=1,
                                         shuffle=False,
                                         pin_memory=True)

    for i, (x, y) in enumerate(loader):
        # Move data to device.
        x = x.to(device)
        y = y.to(device)

        # Compute forward pass.
        pred_y = model(x)

        # Verify shape.
        assert(y.shape[0] == 1)
        assert(len(y.shape) == 2)
        assert(len(pred_y.shape) == 4)

        # Get present classes in image.
        class_idx = np.where(y[0].cpu().data.numpy())[0]

        # Compute a mask for each present class in the image.
        for c in class_idx:
            # Match fully convolutional output shape.
            class_y = torch.zeros_like(pred_y)
            class_y[0, c, :, :] = 1

            # Gradient signal.
            grad_signal = pred_y * class_y

            # TODO: Compute mask.
            pass

if __name__ == '__main__':
    import argparse
    import sys
    import traceback
    try:
        parser = argparse.ArgumentParser(description='Learn perturbation mask')
        parser.add_argument('--data_dir', type=str,
                            default='/datasets/pascal',
                            help='path to root directory containing data')
        parser.add_argument('--checkpoint_path', type=str,
                            default='checkpoint.pth.tar',
                            help='path to save checkpoint')
        parser.add_argument('--arch', type=str, default='vgg16',
                            help='name of CNN architecture (choose from '
                                 'PyTorch pretrained networks')
        parser.add_argument('--dataset', choices=['voc_2007', 'coco_2014'],
                            default='voc_2007',
                            help='name of dataset')
        parser.add_argument('--ann_dir', type=str, default=None,
                            help='path to annotation directory (for COCO).')
        parser.add_argument('--split', choices=['test', 'val2014'],
                            default='test',
                            help='name of split')
        parser.add_argument('--input_size', type=int, default=224,
                            help='CNN image input size')
        parser.add_argument('--gpu', type=int, nargs='*', default=None,
                            help='List of GPU(s) to use.')

        args = parser.parse_args()
        set_gpu(args.gpu)

        learn_masks_for_pointing(args.data_dir,
                                 args.checkpoint_path,
                                 arch=args.arch,
                                 dataset=args.dataset,
                                 ann_dir=args.ann_dir,
                                 split=args.split,
                                 input_size=args.input_size)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
