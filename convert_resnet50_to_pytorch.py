"""
Requires python 2.7 and torch==0.3.1 and pycaffe (CPU is sufficient).
"""
from collections import OrderedDict
import caffe

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

from resnet_caffe import ResNetCaffe, convert_batchnorm, load_resnet50
from utils import replace_module, hook_get_acts


def split(x, delimiter='.'):
    return x.split(delimiter)


def replace_bottleneck(pytorch_model, caffenet_modules, pytorch_name, caffe_name):
    pytorch_model = replace_module(pytorch_model, split('%s.conv1' % pytorch_name), caffenet_modules['res%s_branch2a' % caffe_name])
    pytorch_model = replace_module(pytorch_model, split('%s.bn1' % pytorch_name), nn.Sequential(caffenet_modules['bn%s_branch2a' % caffe_name], caffenet_modules['scale%s_branch2a' % caffe_name]))
    pytorch_model = replace_module(pytorch_model, split('%s.conv2' % pytorch_name), caffenet_modules['res%s_branch2b' % caffe_name])
    pytorch_model = replace_module(pytorch_model, split('%s.bn2' % pytorch_name), nn.Sequential(caffenet_modules['bn%s_branch2b' % caffe_name], caffenet_modules['scale%s_branch2b' % caffe_name]))
    pytorch_model = replace_module(pytorch_model, split('%s.conv3' % pytorch_name), caffenet_modules['res%s_branch2c' % caffe_name])
    pytorch_model = replace_module(pytorch_model, split('%s.bn3' % pytorch_name), nn.Sequential(caffenet_modules['bn%s_branch2c' % caffe_name], caffenet_modules['scale%s_branch2c' % caffe_name]))
    if 'res%s_branch1' % caffe_name in caffenet_modules:
        pytorch_model = replace_module(pytorch_model, split('%s.downsample.0' % pytorch_name), caffenet_modules['res%s_branch1' % caffe_name])
        pytorch_model = replace_module(pytorch_model, split('%s.downsample.1' % pytorch_name), nn.Sequential(caffenet_modules['bn%s_branch1' % caffe_name], caffenet_modules['scale%s_branch1' % caffe_name]))
    return pytorch_model


def compare_tensors(x, y):
    return torch.abs(x - y).sum().data.numpy()[0]


def convert_resnet50_to_pytorch(orig_checkpoint_path, new_checkpoint_path):
    caffenet_model = torch.load(orig_checkpoint_path)
    caffenet_modules = caffenet_model.models
    bottleneck_depths = [3, 4, 6, 3]
    model = ResNetCaffe(models.resnet.Bottleneck, bottleneck_depths, 1000)
    model = convert_batchnorm(model)
    model.eval()

    print(model)

    model = replace_module(model, ['conv1'], caffenet_modules['conv1'])
    model = replace_module(model, ['bn1'],
                           nn.Sequential(caffenet_modules['bn_conv1'],
                                         caffenet_modules['scale_conv1']))

    letter = ord('a')
    bottleneck_names = OrderedDict({})
    for i, d in enumerate(bottleneck_depths):
        for j in range(d):
            bottleneck_names['layer%d.%d' % (i+1, j)] = '%d%s' % (i+2,
                                                                  chr(letter+j))

    for pytorch_name, caffe_name in bottleneck_names.items():
        print('Converting %s to %s' % (caffe_name, pytorch_name))
        model = replace_bottleneck(model,
                                   caffenet_modules,
                                   pytorch_name,
                                   caffe_name)

    # Replace last fully connected layer.
    last_fconv_key = caffenet_modules.keys()[-1]
    assert 'fc8' in last_fconv_key
    caffenet_fconv_sd = caffenet_modules[last_fconv_key].state_dict()
    out_ch, in_ch, h, w = caffenet_fconv_sd['weight'].shape
    assert in_ch == model.fc.in_features
    assert h == 1
    assert w == 1
    new_fc = nn.Linear(in_ch, out_ch)
    new_fc_sd = {
        'weight': caffenet_fconv_sd['weight'].view(out_ch, in_ch),
        'bias': caffenet_fconv_sd['bias']
    }
    new_fc.load_state_dict(new_fc_sd)

    model = replace_module(model, ['fc'], new_fc)
    model.eval()
    caffenet_model.eval()

    print(model)
    print(caffenet_model)
    x = Variable(torch.randn(1, 3, 224, 224))
    caffenet_out = caffenet_model(x)
    pytorch_out = model(x)
    print('Diff in output between caffenet and pytorch models: %.2f'
          % compare_tensors(pytorch_out, caffenet_out['fc8'][:, :, 0, 0]))

    torch.save(model.state_dict(), new_checkpoint_path)

    new_res = load_resnet50(checkpoint_path=new_checkpoint_path)
    new_res.eval()
    new_out = new_res(x)
    print('Diff in output between old and newly loaded pytorch models: %.2f'
          % compare_tensors(pytorch_out, new_out))
    acts1 = hook_get_acts(new_res, bottleneck_names.keys(), x)
    acts2 = hook_get_acts(model, bottleneck_names.keys(), x)
    for i, pytorch_name in enumerate(bottleneck_names.keys()):
        print('Diff at %s: %.2f' % (pytorch_name, compare_tensors(acts1[i], acts2[i])))


if __name__ == '__main__':
    import argparse
    import sys
    import traceback
    try:
        parser = argparse.ArgumentParser(description='Learn perturbation mask')
        parser.add_argument('orig_checkpoint_path', type=str)
        parser.add_argument('new_checkpoint_path', type=str)
        args = parser.parse_args()
        convert_resnet50_to_pytorch(args.orig_checkpoint_path,
                                    args.new_checkpoint_path)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
