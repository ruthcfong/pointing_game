"""
utils.py

Contains helper functions.
"""

from collections import OrderedDict
import copy
import math
import os

import numpy as np

import torch
import torch.nn as nn

from torchvision import models

VOC_CLASSES = np.array([
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
], dtype=str)
COCO_CATEGORY_IDS = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data/coco_category_ids.txt'), dtype=int)


class FromVOCToOneHotEncoding(object):
    def __init__(self, num_classes=20, class_to_idx=None):
        self.num_classes = num_classes
        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(VOC_CLASSES)}
        else:
            self.class_to_idx = class_to_idx
        assert(self.num_classes == len(self.class_to_idx))

    def __call__(self, d):
        assert('annotation' in d)
        assert('object' in d['annotation'])
        objs = d['annotation']['object']
        if isinstance(objs, list):
            classes = [obj['name'] for obj in objs]
        else:
            classes = [objs['name']]

        class_idx = [self.class_to_idx[c] for c in classes]
        label = np.zeros(self.num_classes, dtype=np.float32)
        label[class_idx] = 1
        return label


class FromCocoToOneHotEncoding(object):
    def __init__(self, num_classes=80, class_to_idx=None):
        self.num_classes = num_classes
        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(COCO_CATEGORY_IDS)}
        else:
            self.class_to_idx = class_to_idx
        assert(self.num_classes == len(self.class_to_idx))

    def __call__(self, anns):
        class_idx = []
        label = np.zeros(self.num_classes, dtype=np.float32)
        for ann in anns:
            assert('category_id' in ann)
            class_idx.append(self.class_to_idx[ann['category_id']])
        label[class_idx] = 1
        return label


def blur_input_tensor(tensor, kernel_size=11, sigma=5.0):
    """Blur tensor with a 2D gaussian blur.

    Args:
        tensor: torch.Tensor, 3 or 4D tensor to blur.
        kernel_size: int, size of 2D kernel.
        sigma: float, standard deviation of gaussian kernel.

    Returns:
        4D torch.Tensor that has been smoothed with gaussian kernel.
    """
    ndim = len(tensor.shape)
    if ndim == 3:
        tensor = tensor.unsqueeze(0)
    assert ndim == 4
    num_channels = tensor.shape[1]
    device = tensor.device

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(
        -1*torch.sum((xy_grid - mean)**2., dim=-1) /
        (2.*variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(num_channels, 1, 1, 1)
    gaussian_kernel = gaussian_kernel.to(device)

    padding = nn.ReflectionPad2d(int(mean)).to(device)
    gaussian_filter = nn.Conv2d(in_channels=num_channels,
                                out_channels=num_channels,
                                kernel_size=kernel_size,
                                groups=num_channels,
                                bias=False).to(device)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    smoothed_tensor = gaussian_filter(padding(tensor))

    return smoothed_tensor


class SimpleToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x)


def get_finetune_model(arch='vgg16',
                       dataset='voc_2007',
                       checkpoint_path=None,
                       convert_to_fully_convolutional=False,
                       final_gap_layer=False):
    # Load pre-trained model.
    model = models.__dict__[arch](pretrained=True)
    if arch == 'inception_v3':
        model.aux_logits = False

    # Only fine-tune last layer.
    for p in model.parameters():
        p.requires_grad = False

    # Get the last layer.
    last_name, last_module = list(model.named_modules())[-1]

    # Set number of classes in dataset.
    if 'voc' in dataset:
        num_classes = 20
    elif 'coco' in dataset:
        num_classes = 80
    else:
        assert(False)

    # Construct new last layer.
    if isinstance(last_module, nn.Linear):
        in_features = last_module.in_features
        bias = last_module.bias is not None
        new_layer_module = nn.Linear(in_features, num_classes, bias=bias)
    else:
        assert(False)

    # Replace last layer.
    model = replace_module(model, last_name.split('.'), new_layer_module)

    # Load weights, if provided.
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

    # Convert model to fully convolutional one.
    if convert_to_fully_convolutional:
        model = make_fully_convolutional(model,
                                         final_gap_layer=final_gap_layer)

    # Set model to evaluation mode.
    model.eval()

    return model


def make_fully_convolutional(model, final_gap_layer=False):
    if isinstance(model, models.VGG):
        new_model_layers = list(model.features.children())

        # Get fully-connected layers.
        classifier = list(model.classifier.children())

        # Especially handle first fully-connected layer.
        first_fc = classifier[0].state_dict()
        in_ch = 512
        kernel_size = 7
        out_ch = first_fc['weight'].shape[0]
        orig_in_ch = in_ch * kernel_size * kernel_size
        assert(first_fc['weight'].shape[1] == orig_in_ch)
        first_conv = nn.Conv2d(in_ch, out_ch, (kernel_size, kernel_size))
        first_conv.load_state_dict({
            'weight': first_fc['weight'].view(out_ch,
                                              in_ch,
                                              kernel_size,
                                              kernel_size),
            'bias': first_fc['bias']
        })
        new_model_layers.append(first_conv)

        # Handle subsequent layers.
        for layer in classifier[1:]:
            if isinstance(layer, nn.Linear):
                fc = layer.state_dict()
                out_ch, in_ch = fc['weight'].shape
                conv = nn.Conv2d(in_ch, out_ch, (1, 1))
                conv.load_state_dict({
                    'weight': fc['weight'].view(out_ch, in_ch, 1, 1),
                    'bias': fc['bias']
                })
                new_model_layers.append(conv)
            else:
                new_model_layers.append(layer)
    elif isinstance(model, models.ResNet):
        new_model_layers = list(model.children())[:-1]

        first_fc = model.fc.state_dict()
        out_ch, in_ch = first_fc['weight'].shape
        first_conv = nn.Conv2d(in_ch, out_ch, (1, 1))
        first_conv.load_state_dict({
            'weight': first_fc['weight'].view(out_ch, in_ch, 1, 1),
            'bias': first_fc['bias']
        })

        new_model_layers.append(first_conv)
    else:
        assert(False)

    # Add final global average pooling layer.
    if final_gap_layer and not isinstance(model, models.ResNet):
        new_model_layers.append(nn.AdaptiveAvgPool2d((1, 1)))

    new_model = nn.Sequential(*new_model_layers)

    return new_model


def create_dir_if_necessary(path, is_dir=False):
    """Create directory to path if necessary."""
    parent_dir = get_parent_dir(path) if not is_dir else path
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def get_parent_dir(path):
    """Return parent directory of path."""
    return os.path.abspath(os.path.join(path, os.pardir))


def str2bool(value):
    """Converts string to bool."""
    value = value.lower()
    if value in ('yes', 'true', 't', '1'):
        return True
    if value in ('no', 'false', 'f', '0'):
        return False
    raise ValueError('Boolean argument needs to be true or false. '
                     'Instead, it is %s.' % value)


def get_device():
    """Return torch.device based on if cuda is available."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def get_pytorch_module(net, layer_name):
    """Return PyTorch module."""
    modules = layer_name.split('.')
    if len(modules) == 1:
        return net._modules.get(layer_name)
    else:
        curr_m = net
        for m in modules:
            curr_m = curr_m._modules.get(m)
        return curr_m


def replace_module(parent_module, module_path, replacement_module):
    """Replace a PyTorch module with a replacement module."""
    if isinstance(parent_module, nn.Sequential):
        module_dict = OrderedDict()
    elif isinstance(parent_module, nn.Module):
        new_parent_module = copy.deepcopy(parent_module)
    for (k, v) in parent_module._modules.items():
        if k == module_path[0]:
            if len(module_path) == 1:
                child_module = replacement_module
            else:
                child_module = replace_module(v, module_path[1:],
                                              replacement_module)
        else:
            child_module = v

        if isinstance(parent_module, nn.Sequential):
            module_dict[k] = child_module
        elif isinstance(parent_module, nn.Module):
            setattr(new_parent_module, k, child_module)
        else:
            assert False

    if isinstance(parent_module, nn.Sequential):
        return nn.Sequential(module_dict)
    elif isinstance(parent_module, nn.Module):
        return new_parent_module
    else:
        assert False


def register_hook_on_module(curr_module,
                            module_type,
                            hook_func,
                            hook_direction='backward'):
    """Register hook on all modules of a given type."""
    if isinstance(curr_module, module_type):
        if hook_direction == 'forward':
            curr_module.register_forward_hook(hook_func)
        elif hook_direction == 'backward':
            curr_module.register_backward_hook(hook_func)
        else:
            raise NotImplementedError('Only "forward" and "backward" are '
                                      'supported, not %s.' % hook_direction)
    for m in curr_module.children():
        register_hook_on_module(m,
                                module_type,
                                hook_func,
                                hook_direction=hook_direction)


activations = []

def hook_acts(module, input, output):
    """Forward hook function for saving activations."""
    activations.append(output)


def get_acts(model, input, second_input=None, clone=True):
    """Returns activations saved using existing hooks."""
    del activations[:]
    if second_input is not None:
        _ = model(input, second_input)
    else:
        _ = model(input)
    if clone:
        return [a.clone() for a in activations]
    else:
        return activations


def hook_get_acts(model, layer_names, input, second_input=None, clone=True):
    """Returns activations at specified layers."""
    hooks = []
    for i in range(len(layer_names)):
        hooks.append(
            get_pytorch_module(model, layer_names[i]).register_forward_hook(
                hook_acts))

    acts_res = [a for a in
                get_acts(model, input, second_input=second_input, clone=clone)]

    for h in hooks:
        h.remove()

    return acts_res


def set_gpu(gpu=None):
    """Set visible gpu(s). This function should be called once at beginning.

    Args:
        gpu (NoneType, int, or list of ints): the gpu(s) (zero-indexed) to use;
            None if no gpus should be used.
    Return:
        bool: True if using at least 1 gpu; otherwise False.
    """
    # Check type of gpu.
    if isinstance(gpu, list):
        if gpu:
            for gpu_i in gpu:
                if not isinstance(gpu_i, int):
                    raise ValueError('gpu should be of type NoneType, int, or '
                                     'list of ints. Instead, gpu[%d] is of '
                                     'type %s.' % type(gpu_i))
    elif isinstance(gpu, int):
        pass
    elif gpu is None:
        pass
    else:
        raise ValueError('gpu should be of type NoneType, int, or list of '
                         'ints. Instead, gpu is of type %s.' % type(gpu))

    # Set if gpu usage (i.e., cuda) is enabled.
    if gpu is None:
        cuda = False
    elif isinstance(gpu, list) and not gpu:
        cuda = False
    else:
        cuda = True

    # Set CUDA_VISIBLE_DEVICES environmental variable.
    gpu_params = ''
    if cuda:
        if isinstance(gpu, list):
            gpu_params = str(gpu).strip('[').strip(']')
        else:
            gpu_params = str(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_params

    # Check type of framework.
    num_visible_gpus = torch.cuda.device_count()

    # Check number of visible gpus.
    if isinstance(gpu, list):
        if num_visible_gpus != len(gpu):
            raise ValueError('The following %d gpu(s) should be visible: %s; '
                             'instead, %d gpu(s) are visible.'
                             % (len(gpu), str(gpu), num_visible_gpus))
    elif gpu is None:
        if num_visible_gpus != 0:
            raise ValueError('0 gpus should be visible; instead, %d gpu(s) '
                             'are visible.' % num_visible_gpus)
    else:
        if num_visible_gpus != 1:
            raise ValueError('1 gpu should be visible; instead %d gpu(s) '
                             'are visible.' % num_visible_gpus)
        assert num_visible_gpus == 1

    print("%d GPU(s) being used at the following index(es): %s" % (
        num_visible_gpus, gpu_params))
    return cuda
