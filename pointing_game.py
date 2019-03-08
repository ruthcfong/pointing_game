"""
pointing_game.py

Evaluate visualization method on pointing game evaluation metric.

The pointing game was originally introduced in
Zhang et al., ECCV2016. Top-down Neural Attention by Excitation Backprop.
"""
import time
import numpy as np
from skimage.transform import resize

import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
import torchvision.utils as vutils

import visdom

from utils import (get_finetune_model, VOC_CLASSES, SimpleToTensor, get_device,
                   set_gpu, blur_input_tensor, register_hook_on_module,
                   hook_get_acts, str2bool)


class FromVOCToDenseBoundingBoxes(object):
    """Transformation from VOC annotation dict to Dense Bounding Boxes."""
    def __init__(self, tolerance=0, num_classes=20, class_to_idx=None):
        self.tolerance = tolerance
        self.num_classes = num_classes
        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(VOC_CLASSES)}
        else:
            self.class_to_idx = class_to_idx
        assert(self.num_classes == len(self.class_to_idx))

    def __call__(self, d):
        # Verify annotation dict.
        assert('annotation' in d)
        assert('size' in d['annotation'])
        assert('width' in d['annotation']['size'])
        assert('height' in d['annotation']['size'])
        assert('object' in d['annotation'])

        # Define dense bounding boxes array to be C x H x W.
        height = int(d['annotation']['size']['height'])
        width = int(d['annotation']['size']['width'])
        bboxes = np.zeros((self.num_classes, height, width), dtype=np.float32)

        objs = d['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]

        # For each object, add its bounding box to the dense array.
        for obj in objs:
            assert('name' in obj)
            assert('bndbox' in obj)
            assert(obj['bndbox'].keys()
                   == set(['xmin', 'xmax', 'ymin', 'ymax']))
            # Get object class.
            class_i = self.class_to_idx[obj['name']]

            # Get bounding box coordinates.
            bb = obj['bndbox']

            # Support tolerance margin, as in Zhang et al., ECCV 2016.
            ymin = max(int(bb['ymin'])-1-self.tolerance, 0)
            ymax = min(int(bb['ymax'])-1+self.tolerance, height)
            xmin = max(int(bb['xmin'])-1-self.tolerance, 0)
            xmax = min(int(bb['xmax'])-1+self.tolerance, width)

            # Demark class-specific bounding box in the dense array.
            bboxes[class_i, ymin:ymax+1, xmin:xmax+1] = 1

        return bboxes


class SimpleResize(object):
    """Resize a 3D array, setting the smaller side to the provided size."""
    def __init__(self, size, order=0):
        self.size = size
        self.order = order

    def __call__(self, x):
        assert(isinstance(x, np.ndarray))
        assert(x.ndim == 3)
        c, h, w = x.shape
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return x
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
        else:
            oh = self.size
            ow = int(self.size * w / h)
        new_x = np.transpose(resize(np.transpose(x, (1, 2, 0)), (oh, ow),
                                    order=self.order), (2, 0, 1))
        return new_x


def pointing_game(data_dir,
                  checkpoint_path,
                  arch='vgg16',
                  dataset='voc_2007',
                  input_size=224,
                  vis_method='gradient',
                  tolerance=0,
                  smooth_sigma=0.,
                  debug=False,
                  print_iter=25,
                  eps=1e-6):
    """
    Play the pointing game using a finetuned model and visualization method.

    Args:
        data_dir: String, root directory for dataset.
        checkpoint_path: String, path to model checkpoint.
        arch: String, name of torchvision.models architecture.
        dataset: String, name of dataset.
        input_size: Integer, length of side of the input image.
        vis_method: String, visualization method.
        tolerance: Integer, number of pixels for tolerance margin.
        smooth_sigma: Float, sigma with which to scale Gaussian kernel.
        debug: Boolean, if True, show debug visualizations.
        print_iter: Integer, frequency with which to log messages.
        eps: Float, epsilon value to add to denominator for division.

    Returns:
        (avg_acc, acc): Tuple containing the following:
            avg_acc: Float, pointing game accuracy over all classes,
            acc: ndarray, array containing accuracies for each class.
    """
    if debug:
        viz = visdom.Visdom()

    # Load fine-tuned model with weights and convert to be fully convolutional.
    model = get_finetune_model(arch=arch,
                               dataset=dataset,
                               checkpoint_path=checkpoint_path,
                               convert_to_fully_convolutional=True)

    device = get_device()
    model = model.to(device)

    # 'guided_backprop' as in Springenberg et al., ICLR Workshop 2015.
    if vis_method == 'guided_backprop':
        # Change backwards function for ReLU.
        def guided_hook_function(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)
        register_hook_on_module(curr_module=model,
                                module_type=nn.ReLU,
                                hook_func=guided_hook_function,
                                hook_direction='backward')
    # 'cam' as in Zhou et al., CVPR 2016.
    elif vis_method == 'cam':
        if 'resnet' in arch:
            # Get third to last layer.
            print(list(model.children()))
            layer_name = '%d' % (len(list(model.children())) - 3)
            layer_names = [layer_name]
        else:
            assert(False)
        last_layer = list(model.children())[-1]
        assert(isinstance(last_layer, nn.Conv2d))
        weights = last_layer.state_dict()['weight']
        assert(len(weights.shape) == 4)
        assert(weights.shape[2] == 1 and weights.shape[3] == 1)

    # Prepare data augmentation.
    assert(isinstance(input_size, int))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize,
    ])

    target_transform = transforms.Compose([
        FromVOCToDenseBoundingBoxes(tolerance=tolerance),
        SimpleResize(input_size),
        SimpleToTensor(),
    ])

    if 'voc' in dataset:
        num_classes = 20
        year = dataset.split('_')[-1]
        dset = datasets.VOCDetection(data_dir,
                                     year=year,
                                     image_set='test',
                                     transform=transform,
                                     target_transform=target_transform)
    else:
        assert(False)

    print('Number of examples in dataset split: %d' % len(dset))
    loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False)

    # Prepare to evaluate pointing game.
    hits = np.zeros(num_classes)
    misses = np.zeros(num_classes)
    start = time.time()
    for i, (x, y) in enumerate(loader):
        # Verify shape.
        assert(x.shape[0] == 1)
        assert(y.shape[0] == 1)

        # Move data to device.
        x = x.to(device)
        y = y.to(device)

        # Set input batch size to the number of classes.
        x = x.expand(num_classes, *x.shape[1:])
        x.requires_grad = True

        model.zero_grad()
        pred_y = model(x)

        # Play pointing game using the specified visualization method.
        # 'gradient' is Simonyan et al., ICLR Workshop 2014.
        if vis_method in ['gradient', 'guided_backprop']:

            # Prepare gradient.
            weights = torch.zeros_like(pred_y)
            labels = torch.arange(0, num_classes).to(device)
            labels = labels[:, None, None, None]
            labels_shape = (num_classes, 1, weights.shape[2], weights.shape[3])
            labels = labels.expand(*labels_shape)
            weights.scatter_(1, labels, 1)
            pred_y.backward(weights)

            # Compute gradient visualization.
            vis, _ = torch.max(torch.abs(x.grad), 1, keepdim=True)

            # Smooth gradient visualization as in Zhang et al., ECCV 2016.
            if smooth_sigma > 0:
                vis = blur_input_tensor(vis,
                                        sigma=smooth_sigma*max(vis.shape[2:]))
        elif vis_method == 'cam':
            acts = hook_get_acts(model, layer_names, x)[0]
            vis_lowres = torch.mean(acts * weights, 1, keepdim=True)
            vis = nn.functional.interpolate(vis_lowres,
                                            size=y.shape[2:],
                                            mode='bilinear')
        else:
            assert(False)

        # Get present classes in the image.
        class_idx = np.where(np.sum(y[0].cpu().data.numpy(), (1, 2)) > 0)[0]

        for c in class_idx:
            # Check if maximum point for class-specific visualization is
            # within one of the bounding boxes for that class.
            max_i = torch.argmax(vis[c])
            if y[0,c,:,:].view(-1)[max_i] > 0.5:
                hits[c] += 1
            else:
                misses[c] += 1

        if i % print_iter == 0:
            avg_acc = np.mean(hits / (hits + misses + eps))
            print('[%d/%d] Avg Acc: %.4f Time: %.2f' % (i,
                                                        len(loader),
                                                        avg_acc,
                                                        time.time() - start))
            start = time.time()
            if debug:
                viz.image(vutils.make_grid(x[0].unsqueeze(0), normalize=True),
                          0)
                viz.image(vutils.make_grid(vis, normalize=True), 1)

    acc = hits / (hits + misses)
    avg_acc = np.mean(acc)
    print('Avg Acc: %.4f' % avg_acc)
    return avg_acc, acc


if __name__ == '__main__':
    import argparse
    import sys
    import traceback
    try:
        parser = argparse.ArgumentParser(description='Learn perturbation mask')
        parser.register('type', 'bool', str2bool)
        parser.add_argument('--data_dir', type=str,
                            default='/datasets/pascal',
                            help='path to root directory containing data')
        parser.add_argument('--checkpoint_path', type=str,
                            default='checkpoint.pth.tar',
                            help='path to save checkpoint')
        parser.add_argument('--arch', type=str, default='vgg16',
                            help='name of CNN architecture (choose from '
                                 'PyTorch pretrained networks')
        parser.add_argument('--dataset', choices=['voc_2007'],
                            default='voc_2007',
                            help='name of dataset')
        parser.add_argument('--input_size', type=int, default=224,
                            help='CNN image input size')
        parser.add_argument('--vis_method', type=str,
                            choices=['gradient', 'guided_backprop', 'cam'],
                            default='gradient',
                            help='CNN image input size')
        parser.add_argument('--tolerance', type=int, default=0,
                            help='amount of tolerance to add')
        parser.add_argument('--smooth_sigma', type=float, default=0.,
                            help='amount of Gaussian smoothing to apply')
        parser.add_argument('--gpu', type=int, nargs='*', default=None,
                            help='List of GPU(s) to use.')
        parser.add_argument('--debug', type='bool', default=False)

        args = parser.parse_args()
        set_gpu(args.gpu)
        pointing_game(args.data_dir,
                      args.checkpoint_path,
                      arch=args.arch,
                      dataset=args.dataset,
                      input_size=args.input_size,
                      vis_method=args.vis_method,
                      tolerance=args.tolerance,
                      smooth_sigma=args.smooth_sigma,
                      debug=args.debug)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
