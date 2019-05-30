"""
pointing_game.py

Evaluate visualization method on pointing game evaluation metric.

The pointing game was originally introduced in
Zhang et al., ECCV2016. Top-down Neural Attention by Excitation Backprop.
"""
import os
import time

import cv2
import numpy as np

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from skimage.transform import resize

from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
import torchvision.utils as vutils

import tqdm

import visdom

from caffe_transforms import get_caffe_transform, CaffeChannelSwap

from utils import (get_finetune_model, VOC_CLASSES, SimpleToTensor, get_device,
                   set_gpu, blur_input_tensor, register_hook_on_module,
                   hook_get_acts, str2bool, COCO_CATEGORY_IDS, RISE,
                   create_dir_if_necessary, get_pytorch_module)

from compute_score import compute_metrics

MAX_GPU_LENGTH = 500


def get_synset(class_ix):
    synset = np.loadtxt('./data/synset_words.txt', dtype=str, delimiter='\t')
    return synset[class_ix][0].split(' ')[0]

class NumpyToTensor(object):
    def __call__(self, img):
        assert(img.ndim == 3)
        x = torch.from_numpy(img)
        x = x.transpose(0, 1).transpose(0, 2).contiguous()
        return x


class RGBtoBGR(object):
    """Convert image from RGB to BGR."""
    def __call__(self, img):
        assert(isinstance(img, Image.Image))
        assert(img.mode == 'RGB')
        rgb_arr = np.asarray(img)
        assert(rgb_arr.ndim == 3)
        assert(rgb_arr.shape[2] == 3)
        r = rgb_arr[:, :, 0]
        g = rgb_arr[:, :, 1]
        b = rgb_arr[:, :, 2]
        bgr_arr = np.zeros_like(rgb_arr)
        bgr_arr[:, :, 0] = b
        bgr_arr[:, :, 1] = g
        bgr_arr[:, :, 2] = r
        assert(np.all(bgr_arr[:,:,0] == b))
        assert(np.all(bgr_arr[:,:,1] == g))
        assert(np.all(bgr_arr[:,:,2] == r))
        return bgr_arr.astype(np.float32)


class FromCocoToDenseSegmentationMasks(object):
    """Transformation from list of COCO annotation dicts to dense segmentation masks."""
    def __init__(self, coco, tolerance=0, backup_size=224, num_classes=80, class_to_idx=None):
        self.coco = coco
        self.tolerance = tolerance
        if self.tolerance > 0:
            self.kernel = np.ones((self.tolerance*2+1, self.tolerance*2+1),
                                  dtype=np.uint8)
        self.backup_size = backup_size
        self.num_classes = num_classes
        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(COCO_CATEGORY_IDS)}
        else:
            self.class_to_idx = class_to_idx
        assert(self.num_classes == len(self.class_to_idx))

    def __call__(self, anns):
        if len(anns) == 0:
            return np.zeros((self.num_classes,
                             self.backup_size,
                             self.backup_size), dtype=np.float32)

        mask = self.coco.annToMask(anns[0])
        seg_masks = np.zeros((mask.shape[0], mask.shape[1], self.num_classes),
                             dtype=np.float32)
        for ann in anns:
            assert('category_id' in ann)
            class_i = self.class_to_idx[ann['category_id']]
            mask = self.coco.annToMask(ann)
            seg_masks[:, :, class_i] += mask

        seg_masks = (seg_masks > 0).astype(np.float32)

        if self.tolerance > 0:
            seg_masks = cv2.dilate(seg_masks, self.kernel, iterations=1)
        seg_masks = np.transpose(seg_masks, (2, 0, 1))

        return seg_masks


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
            assert(obj['difficult'] in ['0','1'])
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


class SortedFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(SortedFolder, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        names = [os.path.basename(x[0]) for x in self.samples]
        self.sorted_idx = np.argsort(names)
        new_samples = [self.samples[i] for i in self.sorted_idx]
        self.samples = new_samples


def pointing_game(data_dir,
                  checkpoint_path,
                  out_path=None,
                  save_dir=None,
                  load_from_save_dir=False,
                  arch='vgg16',
                  converted_caffe=False,
                  dataset='voc_2007',
                  ann_dir=None,
                  split='test',
                  metric='pointing',
                  input_size=224,
                  vis_method='gradient',
                  final_gap_layer=False,
                  debug=False,
                  print_iter=1,
                  save_iter=25,
                  start_index=-1,
                  end_index=-1,
                  layer_name=None,
                  eps=1e-6,
                  num_masks=4000,
                  s=7,
                  p1=0.5,
                  rise_filter_path=None,
                  gpu_batch=100):
    """
    Play the pointing game using a finetuned model and visualization method.

    Args:
        data_dir: String, root directory for dataset.
        checkpoint_path: String, path to model checkpoint.
        out_path: String, path to save per-image results to.
        save_dir: String, path to directory to save per-image visualizations.
        arch: String, name of torchvision.models architecture.
        converted_caffe: Boolean, if True, use weights converted from Caffe.
        dataset: String, name of dataset.
        ann_dir: String, path to root directory containing annotation files
            (used for COCO).
        split: String, name of split to use for evaluation.
        input_size: Integer, length of side of the input image.
        vis_method: String, visualization method.
        tolerance: Integer, number of pixels for tolerance margin.
        smooth_sigma: Float, sigma with which to scale Gaussian kernel.
        final_gap_layer: Boolean, if True, add a final gap layer.
        debug: Boolean, if True, show debug visualizations.
        print_iter: Integer, frequency with which to log messages.
        eps: Float, epsilon value to add to denominator for division.

    Returns:
        (avg_acc, acc): Tuple containing the following:
            avg_acc: Float, pointing game accuracy over all classes,
            acc: ndarray, array containing accuracies for each class.
    """
    if metric == 'pointing':
        tolerance = 15
    elif metric == 'average_precision':
        tolerance = 0
    else:
        assert False

    if vis_method in ['gradient', 'guided_backprop']:
        smooth_sigma = 0.02
    else:
        smooth_sigma = 0

    if debug:
        viz = visdom.Visdom(env=f'pointing_caffe_{converted_caffe}')

    # Load fine-tuned model with weights and convert to be fully convolutional.
    model = get_finetune_model(arch=arch,
                               dataset=dataset,
                               converted_caffe=converted_caffe,
                               checkpoint_path=checkpoint_path,
                               convert_to_fully_convolutional=True,
                               final_gap_layer=final_gap_layer)

    # Handle large images on CPU.
    cpu_device = torch.device('cpu')

    # Handle all other images on GPU, if available.
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
            layer_name = '%d' % (len(list(model.children())) - 3)
            layer_names = [layer_name]
        elif 'googlenet' in arch:
            # Get second to last layer (exclude GAP and last fc layer).
            layer_name = '%d' % (len(list(model.children())) - 2)
            layer_names = [layer_name]
        else:
            assert False
        last_layer = list(model.children())[-1]
        assert(isinstance(last_layer, nn.Conv2d))
        weights = last_layer.state_dict()['weight']
        assert(len(weights.shape) == 4)
        assert(weights.shape[2] == 1 and weights.shape[3] == 1)
    elif vis_method == 'grad_cam':
        if 'vgg16' in arch:
            if layer_name is not None:
                layer_names = [layer_name]
            else:
                layer_names = ['29'] # last conv layer in features (pre-pooling) (14 x 14)
        elif 'resnet50' in arch:
            if layer_name is not None:
                layer_names = [layer_name]
            else:
                layer_names = ['7'] # last conv layer before GAP and FC layer.
        elif 'googlenet' in arch:
            if layer_name is not None:
                layer_names = [layer_name]
            else:
                layer_names = ['15'] # last conv layer before GAP and FC layer.
        else:
            assert(False)
        # Prepare to get backpropagated gradient at intermediate layer.
        grads = []
        def hook_grads(module, grad_in, grad_out):
            grads.append(grad_in)
        assert(len(layer_names) == 1)
        hook = get_pytorch_module(model, layer_names[0]).register_backward_hook(hook_grads)
    elif vis_method == 'rise':
        explainer = RISE(model, input_size, device=device, gpu_batch=gpu_batch)
        if rise_filter_path is None or not os.path.exists(rise_filter_path):
            if rise_filter_path is None:
                rise_filter_path = 'masks.npy'
            create_dir_if_necessary(rise_filter_path)
            explainer.generate_masks(N=num_masks, s=s, p1=p1, savepath=rise_filter_path)
        else:
            explainer.load_masks(filepath=rise_filter_path)

    # Prepare data augmentation.
    assert(isinstance(input_size, int))
    if vis_method == 'rise':
        resize_transform = transforms.Resize((input_size, input_size))
    else:
        resize_transform = transforms.Resize(input_size)
    if converted_caffe:
        if vis_method == 'rise':
            transform = get_caffe_transform(size=(input_size, input_size))
        else:
            transform = get_caffe_transform(size=input_size)
    else:
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            normalize_transform,
        ])

    if 'voc' in dataset:
        target_transform = transforms.Compose([
            FromVOCToDenseBoundingBoxes(tolerance=tolerance),
            SimpleResize(input_size),
            SimpleToTensor(),
        ])

        num_classes = 20
        year = dataset.split('_')[-1]
        dset = datasets.VOCDetection(data_dir,
                                     year=year,
                                     image_set=split,
                                     transform=transform,
                                     target_transform=target_transform)
    elif 'coco' in dataset:
        num_classes = 80
        print(ann_dir)
        ann_path = os.path.join(ann_dir, 'instances_%s.json' % split)

        dset = datasets.CocoDetection(os.path.join(data_dir, split),
                                      ann_path,
                                      transform=transform,
                                      target_transform=None)

        target_transform = transforms.Compose([
            FromCocoToDenseSegmentationMasks(dset.coco, tolerance=tolerance),
            SimpleResize(input_size),
            SimpleToTensor(),
        ])

        dset.target_transform = target_transform
    elif 'imnet' in dataset:
        num_classes = 1000
        dset = SortedFolder(os.path.join(data_dir, split), transform=transform)
    else:
        assert(False)

    print(f'dataset: {dataset}\n'
          f'split: {split}\n'
          f'arch: {arch}\n'
          f'metric: {metric}\n'
          f'smooth_sigma: {smooth_sigma}\n'
          f'tolerance: {tolerance}\n' 
          f'checkpoint_path: {checkpoint_path}\n'
          f'out_path: {out_path}\n'
          f'save_dir: {save_dir}\n')
    print('Number of examples in dataset split: %d' % len(dset))
    if start_index != -1 or end_index != -1:
        if end_index == -1:
            end_index = len(dset)
        if start_index == -1:
            start_index = 0
        idx = range(start_index, end_index)
        dset = torch.utils.data.Subset(dset, idx)
        print(f'Evaluating from {start_index} to {end_index}')
    else:
        start_index = 0
        end_index = len(dset)

    if save_dir is not None:
        create_dir_if_necessary(save_dir, is_dir=True)

    # Prepare to evaluate pointing game.
    if out_path is not None:
        if os.path.exists(out_path):
            print('Loading previous records...')
            records = np.loadtxt(out_path)
        else:
            records = np.zeros((len(dset), num_classes))
    if metric == 'pointing':
        hits = np.zeros(num_classes)
        misses = np.zeros(num_classes)
    elif metric == 'average_precision':
        sum_precs = np.zeros(num_classes)
        num_examples = np.zeros(num_classes)

    if out_path is not None and np.sum(records) != 0:
        if metric == 'pointing':
            hits = np.sum(records == 1, 0)
            misses = np.sum(records == -1, 0)
        elif metric == 'average_precision':
            sum_precs = np.sum(records, 0)
            num_examples = np.sum(records != 0, 0)
        next_index = np.where(records != 0)[0][-1]+1
        print(f'Next Index {next_index}')
    else:
        next_index = 0

    if next_index > 0:
        dset = torch.utils.data.Subset(dset, range(next_index, len(dset)))
        start_index = next_index

    loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False)

    image_idx = []
    y_shapes = []
    vis_shapes = []
    using_cpu = False
    t_loop = tqdm.tqdm(loader)
    for i, (x, y) in enumerate(t_loop):
        if (save_dir is not None
            and os.path.exists(os.path.join(save_dir, f'{i+start_index:06d}.pth'))
            and 'imnet' in dataset):
            print(f'Skipping image {i+start_index}; already saved.')
            continue

        # Verify shape.
        assert(x.shape[0] == 1)
        assert(y.shape[0] == 1)

        # Move data to device.
        x = x.to(device)

        # Get present classes in the image.
        if 'imnet' in dataset:
            class_idx = y.numpy()
        else:
            class_idx = np.where(np.sum(y[0].cpu().data.numpy(), (1, 2)) > 0)[0]
        curr_num_classes = len(class_idx)
        if curr_num_classes == 0:
            print(f'Skipping image {i+start_index}; no classes in it.')
            continue
        assert(curr_num_classes >= 1)

        if vis_method != 'rise':
            # Set input batch size to the number of classes.
            x = x.expand(curr_num_classes, *x.shape[1:])
            if vis_method is not 'cam':
                x.requires_grad = True

            model.zero_grad()
            try:
                pred_y = model(x)
            except RuntimeError:
                using_cpu = True
                print(f'Using CPU to handle image {i+start_index} with shape {x.shape}.')
                # x = torch.tensor(x, device=cpu_device, requires_grad=True)
                x = x.cpu().clone().detach().requires_grad_(True)
                model.to(cpu_device)
                model.zero_grad()
                pred_y = model(x)

        # Play pointing game using the specified visualization method.
        # 'gradient' is Simonyan et al., ICLR Workshop 2014.
        if vis_method in ['gradient', 'guided_backprop']:

            # Prepare gradient.
            weights = torch.zeros_like(pred_y)
            labels = torch.from_numpy(class_idx).to(pred_y.device)
            labels = labels[:, None, None, None]
            labels_shape = (curr_num_classes, 1, weights.shape[2], weights.shape[3])
            labels = labels.expand(*labels_shape)
            weights.scatter_(1, labels, 1)
            try:
                pred_y.backward(weights)
            except RuntimeError:
                # TODO(ruthfong): Handle with less redundancy.
                using_cpu = True
                print(f'Using CPU to handle image {i+start_index} with shape {x.shape}.')
                # x = torch.tensor(x, device=cpu_device, requires_grad=True)
                x = x.cpu().clone().detach().requires_grad_(True)
                model.to(cpu_device)
                model.zero_grad()
                pred_y = model(x)

                weights = torch.zeros_like(pred_y)
                labels = torch.from_numpy(class_idx).to(pred_y.device)
                labels = labels[:, None, None, None]
                labels_shape = (curr_num_classes, 1, weights.shape[2], weights.shape[3])
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
            try:
                acts = hook_get_acts(model, layer_names, x)[0]
            except RuntimeError:
                using_cpu = True
                print(f'Using CPU to handle image {i+start_index} with shape {x.shape}.')
                x = x.cpu().clone().detach().requires_grad_(True)
                model.to(cpu_device)
                acts = hook_get_acts(model, layer_names, x)[0]

            vis_lowres = torch.mean(acts * weights[class_idx].to(acts.device),
                                    1,
                                    keepdim=True)
            vis = nn.functional.interpolate(vis_lowres,
                                            size=y.shape[2:],
                                            mode='bilinear')
        elif vis_method == 'grad_cam':
            # Prepare gradient.
            weights = torch.zeros_like(pred_y)
            labels = torch.from_numpy(class_idx).to(pred_y.device)
            labels = labels[:, None, None, None]
            labels_shape = (curr_num_classes, 1, weights.shape[2], weights.shape[3])
            labels = labels.expand(*labels_shape)
            weights.scatter_(1, labels, 1)

            # Get backpropagated gradient at intermediate layer.
            try:
                pred_y.backward(weights)
            except:
                # TODO(ruthfong): Handle with less redundancy.
                using_cpu = True
                print(f'Using CPU to handle image {i+start_index} with shape {x.shape}.')
                # x = torch.tensor(x, device=cpu_device, requires_grad=True)
                x = x.cpu().clone().detach().requires_grad_(True)
                model.to(cpu_device)
                model.zero_grad()
                pred_y = model(x)

                weights = torch.zeros_like(pred_y)
                labels = torch.from_numpy(class_idx).to(pred_y.device)
                labels = labels[:, None, None, None]
                labels_shape = (curr_num_classes, 1, weights.shape[2], weights.shape[3])
                labels = labels.expand(*labels_shape)
                weights.scatter_(1, labels, 1)
                pred_y.backward(weights)

            assert (len(grads) == 1)
            if len(grads[0]) == 1:
                grad = grads[0][0]
            else:
                assert('googlenet' in arch)
                grad = torch.cat(grads[0], 1)
            grad = grad.to(pred_y.device)
            # assert(len(grads[0]) == 1)
            #grad = grads[0][-1]
            del grads[:]
            #hook.remove()

            # Get activations at intermediate layer.
            acts = hook_get_acts(model, layer_names, x)[0]

            # Apply global average pooling to intermediate gradient.
            grad_weights = torch.mean(grad, (2, 3), keepdim=True)

            # Linearly combine activations and gradient weights.
            try:
                grad_cam = torch.sum(acts * grad_weights, 1, keepdim=True)
            except:
                import pdb; pdb.set_trace()

            # Apply ReLU to GradCAM vis.
            vis_lowres = torch.clamp(grad_cam, min=0)

            # Upsample visualization to image size.
            vis = nn.functional.interpolate(vis_lowres,
                                            size=x.shape[2:],
                                            mode='bilinear')

        elif vis_method == 'rise':
            if load_from_save_dir:
                try:
                    vis = torch.load(os.path.join(save_dir, f'{i+start_index:06d}.pth'))
                    if isinstance(vis, torch.Tensor):
                        vis = vis[class_idx]
                    else:
                        assert isinstance(vis, dict)
                        vis = vis['vis']
                        assert np.all(class_idx == vis['class_idx'])
                except:
                    print(f'No file for {i+start_index:06d}, running RISE.')
                    vis = explainer(x)
                    vis = vis.unsqueeze(1)
                    # Upsample visualization to image size.
                    vis = nn.functional.interpolate(vis,
                                                    size=y.shape[2:],
                                                    mode='bilinear')
                    torch.save(vis, os.path.join(save_dir, f'{i+start_index:06d}.pth'))

            else:
                vis = explainer(x)
                vis = vis[class_idx]
                vis = vis.unsqueeze(1)
                # Upsample visualization to image size.
                if not 'imnet' in dataset:
                    vis = nn.functional.interpolate(vis,
                                                    size=y.shape[2:],
                                                    mode='bilinear')
        else:
            assert(False)

        if save_dir is not None and not load_from_save_dir:
            save_path = os.path.join(save_dir, get_synset(class_idx))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            torch.save({'mask': vis,
                        'class_idx': class_idx,
                        }, os.path.join(save_path, 'ILSVRC2012_val_' + f'{i+1+start_index:08d}.JPEG.pth'))

        if 'imnet' in dataset:
            continue

    
        # Move model back to GPU if necessary.
        if using_cpu:
            model.to(device)
            using_cpu = False

        y_shape = y.shape[2:]
        vis_shape = vis.shape[2:]
        image_idx.append(i+start_index)
        y_shapes.append(y_shape)
        vis_shapes.append(vis_shape)
        if y.shape[2] != vis.shape[2] or y.shape[3] != vis.shape[3]:
            print(f'{i+start_index:06d}: output shape {y_shape} and vis shape {vis_shape} do not match')
            continue

        for class_i, c in enumerate(class_idx):
            # Check if maximum point for class-specific visualization is
            # within one of the bounding boxes for that class.
            if metric == 'pointing':
                max_i = torch.argmax(vis[class_i])
                if y[0,c,:,:].view(-1)[max_i] > 0.5:
                    hits[c] += 1
                    if out_path is not None:
                        records[i,c] = 1
                else:
                    misses[c] += 1
                    if out_path is not None:
                        records[i,c] = -1
            elif metric == 'average_precision':
                # Flatten visualization and ground truth data.
                y_flat = y[0,c].reshape(-1).float()
                vis_flat = vis[class_i].reshape(-1).cpu().data.numpy()
                ap = average_precision_score(y_flat, vis_flat)
                sum_precs[c] += ap
                num_examples[c] += 1
                if out_path is not None:
                    records[i,c] = ap
            else:
                assert(False)
            if debug:
                def normalize_arr(x):
                    x_min, x_max = np.min(x), np.max(x)
                    return (x - x_min) / (x_max - x_min)

                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                if converted_caffe:
                    viz.image(vutils.make_grid(CaffeChannelSwap()(x[0]).unsqueeze(0), normalize=True), win=0)
                else:
                    viz.image(vutils.make_grid(x, normalize=True), win=0)
                viz.image(vutils.make_grid(vis[class_i], normalize=True), win=1)
                # time.sleep(1)
                f, ax = plt.subplots(1, 1)
                if converted_caffe:
                    ax.imshow(vutils.make_grid(CaffeChannelSwap()(x[0]).unsqueeze(0), normalize=True).cpu().data.squeeze().numpy().transpose(1, 2, 0))
                else:
                    ax.imshow(vutils.make_grid(x, normalize=True).cpu().data.squeeze().numpy().transpose(1, 2, 0))
                ax.imshow(resize(normalize_arr(vis[class_i].cpu().data.numpy().transpose(1, 2, 0)), x.shape[2:]).squeeze(), alpha=0.5, cmap='jet')
                create_dir_if_necessary(os.path.join(save_dir, 'debug_images'), True)
                plt.savefig(os.path.join(save_dir, 'debug_images', f'{i+start_index:06d}_class_{class_i}.png'))
                plt.close()
                # print(np.argmax(y[0].cpu().data.numpy()))

        if i % print_iter == 0:
            if metric == 'pointing':
                running_avg = np.mean(hits / (hits + misses + eps))
                metric_name = 'Avg Acc'
            elif metric == 'average_precision':
                running_avg = np.mean(sum_precs / (num_examples + eps))
                metric_name = 'Mean Avg Prec'
            t_loop.set_description(f'{metric_name} {running_avg:.4f}')
            if debug:
                pass
                viz.image(vutils.make_grid(x[0].unsqueeze(0), normalize=True), 0)
                viz.image(vutils.make_grid(vis, normalize=True), 1)
        if i % save_iter == 0 and out_path is not None:
            create_dir_if_necessary(out_path)
            np.savetxt(out_path, records)
            torch.save({'image_idx': image_idx,
                        'vis_shapes': vis_shapes,
                        'y_shapes': y_shapes}, f'errors_new_v2_{vis_method}_{dataset}_{metric}.pth')

    torch.save({'image_idx': image_idx,
                'vis_shapes': vis_shapes,
                'y_shapes': y_shapes}, f'errors_new_v2_{vis_method}_{dataset}_{metric}.pth')
    if out_path is not None:
        create_dir_if_necessary(out_path)
        np.savetxt(out_path, records)
    if metric == 'pointing':
        acc = hits / (hits + misses)
        avg_acc = np.mean(acc)
        print('Avg Acc: %.4f' % avg_acc)
        print(acc)
        return avg_acc, acc
    elif metric == 'average_precision':
        class_mean_avg_prec = sum_precs / num_examples
        mean_avg_prec = np.mean(class_mean_avg_prec)
        print('Mean Avg Prec: %.4f' % mean_avg_prec)
        print(class_mean_avg_prec)
        return mean_avg_prec, class_mean_avg_prec

    if out_path is not None:
        compute_metrics(out_path, metric=metric, dataset=dataset)
    

def find_best_alpha(data_dir,
                    checkpoint_path,
                    out_prefix=None,
                    arch='vgg16',
                    dataset='voc_2007',
                    ann_dir=None,
                    split='test',
                    threshold_type='mean',
                    input_size=224,
                    vis_method='gradient',
                    tolerance=0,
                    smooth_sigma=0.,
                    final_gap_layer=False):
    metric = 'average_precision'
    if threshold_type == 'mean':
        alphas = np.arange(0, 10.5, 0.5)
    elif threshold_type == 'min_max_diff':
        alphas = np.arange(0, 1, 0.05)
    elif threshold_type == 'energy':
        alphas = np.arange(0, 1, 0.05)
    else:
        assert(False)

    best_alpha = -1
    best_map = 0
    maps = np.zeros(len(alphas))
    best_class_maps = 0
    for i, alpha in enumerate(alphas):
        if out_prefix is not None:
            out_path = f'{out_prefix}_alpha_{alpha:.2f}.txt'
        else:
            out_path = None
        map, class_map = pointing_game(data_dir,
                                       checkpoint_path,
                                       out_path=out_path,
                                       arch=arch,
                                       dataset=dataset,
                                       ann_dir=ann_dir,
                                       split=split,
                                       metric=metric,
                                       threshold_type=threshold_type,
                                       input_size=input_size,
                                       vis_method=vis_method,
                                       tolerance=tolerance,
                                       smooth_sigma=smooth_sigma,
                                       final_gap_layer=final_gap_layer,
                                       alpha=alpha)
        maps[i] = map
        if map > best_map:
            best_alpha = alpha
            best_map = map
            best_class_maps = class_map
    print(f'Best Alpha for {threshold_type} on {dataset} {split} using {arch} '
          f'and {vis_method}: {best_alpha:.2f} with MAP {best_map:.4f}')
    print(best_alpha)
    print(best_class_maps)


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
                            default=None,
                            help='path to save checkpoint')
        parser.add_argument('--arch', type=str, default='vgg16',
                            help='name of CNN architecture (choose from '
                                 'PyTorch pretrained networks')
        parser.add_argument('--dataset',
                            choices=['voc_2007', 'coco_2014', 'coco_2017', 'imnet'],
                            default='voc_2007',
                            help='name of dataset')
        parser.add_argument('--ann_dir', type=str, default=None,
                            help='path to root directory containing '
                                 'annotation files (for COCO).')
        parser.add_argument('--split', type=str,
                            choices=['val', 'test', 'val2014', 'val2017', 'val_pytorch'],
                            default='test',
                            help='name of split to use')
        parser.add_argument('--input_size', type=int, default=224,
                            help='CNN image input size')
        parser.add_argument('--vis_method', type=str,
                            choices=['gradient', 'guided_backprop', 'cam',
                                     'grad_cam', 'rise'],
                            default='gradient',
                            help='CNN image input size')
        parser.add_argument('--final_gap_layer', type='bool', default=True,
                            help='if True, add a final GAP layer')
        parser.add_argument('--gpu', type=int, nargs='*', default=None,
                            help='List of GPU(s) to use.')
        parser.add_argument('--debug', type='bool', default=False)
        parser.add_argument('--metric', type=str, choices=['pointing',
                                                           'average_precision'],
                            default='pointing')
        parser.add_argument('--converted_caffe', type='bool', default=False)
        parser.add_argument('--out_path', type=str, default=None)
        parser.add_argument('--save_dir', type=str, default=None)
        parser.add_argument('--start_index', type=int, default=-1)
        parser.add_argument('--end_index', type=int, default=-1)
        parser.add_argument('--load_from_save_dir', type='bool', default=False)
        parser.add_argument('--layer_name', type=str, default=None)
        parser.add_argument('--num_masks', type=int, default=4000)
        parser.add_argument('--p1', type=float, default=0.5)
        parser.add_argument('--s', type=int, default=7)
        parser.add_argument('--rise_filter_path', type=str, default='masks.npy')
        parser.add_argument('--gpu_batch', type=int, default=100)

        args = parser.parse_args()
        set_gpu(args.gpu)

        pointing_game(args.data_dir,
                      args.checkpoint_path,
                      out_path=args.out_path,
                      save_dir=args.save_dir,
                      load_from_save_dir=args.load_from_save_dir,
                      arch=args.arch,
                      converted_caffe=args.converted_caffe,
                      dataset=args.dataset,
                      ann_dir=args.ann_dir,
                      split=args.split,
                      metric=args.metric,
                      input_size=args.input_size,
                      vis_method=args.vis_method,
                      final_gap_layer=args.final_gap_layer,
                      start_index=args.start_index,
                      end_index=args.end_index,
                      layer_name=args.layer_name,
                      debug=args.debug,
                      num_masks=args.num_masks,
                      p1=args.p1,
                      s=args.s,
                      rise_filter_path=args.rise_filter_path,
                      gpu_batch=args.gpu_batch)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
