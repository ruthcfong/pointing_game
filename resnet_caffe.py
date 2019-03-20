import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from torchvision import models



class ResNetCaffe(models.resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNetCaffe, self).__init__(block, layers, num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
                                    ceil_mode=True)
        for i in range(2, 5):
            getattr(self, 'layer%d'%i)[0].conv1.stride = (2,2)
            getattr(self, 'layer%d'%i)[0].conv2.stride = (1,1)


class Scale(nn.Module):
    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = Parameter(torch.Tensor(channels))
        self.bias = Parameter(torch.Tensor(channels))
        self.channels = channels

    def __repr__(self):
        return 'Scale(channels = %d)' % self.channels

    def forward(self, x):
        nb, nc, nh, nw = x.shape
        x = x * self.weight.view(1, nc, 1, 1)
        x = x + self.bias.view(1, nc, 1, 1)
        return x


def replace_every_module(parent_module, orig_module_class, replace_func):
    if isinstance(parent_module, nn.Sequential):
        module_dict = OrderedDict()
    elif isinstance(parent_module, nn.Module):
        new_parent_module = copy.deepcopy(parent_module)
    else:
        assert (False)
    for (k, v) in parent_module._modules.items():
        # print v
        if isinstance(v, orig_module_class):
            child_module = replace_func(v)
        elif len(v._modules.items()) > 0:
            child_module = replace_every_module(v,
                                                orig_module_class,
                                                replace_func)
        else:
            child_module = v

        if isinstance(parent_module, nn.Sequential):
            module_dict[k] = child_module
        elif isinstance(parent_module, nn.Module):
            setattr(new_parent_module, k, child_module)

    if isinstance(parent_module, nn.Sequential):
        return nn.Sequential(module_dict)
    elif isinstance(parent_module, nn.Module):
        return new_parent_module


def batchnorm_replace_func(x):
    assert isinstance(x, nn.BatchNorm2d)
    num_features = x.num_features
    new_batchnorm = nn.BatchNorm2d(num_features, momentum=0.9, affine=False)
    scale = Scale(num_features)
    return nn.Sequential(new_batchnorm, scale)


def convert_batchnorm(model):
    model = replace_every_module(model, nn.BatchNorm2d, batchnorm_replace_func)
    return model


def load_resnet50(checkpoint_path=None):
    state_dict = torch.load(checkpoint_path)
    assert 'fc.weight' in state_dict
    num_classes, _ = state_dict['fc.weight'].shape
    model = ResNetCaffe(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes)
    model = convert_batchnorm(model)
    model.load_state_dict(state_dict)
    model.eval()
    return model
