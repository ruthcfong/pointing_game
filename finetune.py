"""
finetune.py

Finetune a network pre-trained on ImageNet for VOC2007.
"""
import os
import shutil
import time

import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

from utils import (set_gpu, get_device, create_dir_if_necessary,
                   get_finetune_model, SimpleToTensor, str2bool,
                   FromVOCToOneHotEncoding, FromCocoToOneHotEncoding)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def precision_and_recall(output, target, tp=0, fp=0, tn=0, fn=0):
    """Compute precision and recall."""
    tp += torch.sum((output > 0) & (target > 0.5))
    fp += torch.sum((output > 0) & (target <= 0.5))
    tn += torch.sum((output <= 0) & (target <= 0.5))
    fn += torch.sum((output <= 0) & (target > 0.5))
    prec = tp / (fp + tp).float()
    rec = tp / (tp + fn).float()
    return prec, rec, tp, fp, tn, fn


def train(train_loader, model, criterion, optimizer, epoch, device,
          print_freq=25):
    """
    Train model for one epoch.

    Args:
        train_loader: torch DataLoader.
        model:  torch.nn.Module, model to train.
        criterion: torch.nn.Module, loss criterion function.
        optimizer: torch.optim.Optimizer,
        epoch: Integer, current epoch.
        device: torch.device, device to load model and data to.
        print_freq: Integer, frequency with which to print log messages.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    precs = AverageMeter()
    recs = AverageMeter()
    tp, fp, tn, fn = 0, 0, 0, 0

    # Switch to train mode.
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # Measure data loading time.
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        # Compute output.
        output = model(input)
        loss = criterion(output, target)

        # Measure precision and recall and record loss.
        prec, rec, tp, fp, tn, fn = precision_and_recall(output, target,
                                                         tp, fp, tn, fn)
        losses.update(loss.item(), input.size(0))
        precs.update(prec.item(), input.size(0))
        recs.update(rec.item(), input.size(0))

        # Compute gradient and do SGD step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {precs.val:.3f}\t'
                  'Rec {recs.val:.3f}\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, precs=precs, recs=recs))


def validate(val_loader, model, criterion, device, print_freq=25):
    """
    Validate model for one epoch.

    Args:
        val_loader: torch DataLoader.
        model:  torch.nn.Module, model to train.
        criterion: torch.nn.Module, loss criterion function.
        device: torch.device, device to load model and data to.
        print_freq: Integer, frequency with which to print log messages.

    Returns:
        losses.avg: Average loss for the current epoch.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    precs = AverageMeter()
    recs = AverageMeter()
    tp, fp, tn, fn = 0, 0, 0, 0

    # Switch to evaluate mode.
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # Compute output.
            output = model(input)
            loss = criterion(output, target)

            # Measure precision and recall and record loss.
            prec, rec, tp, fp, tn, fn = precision_and_recall(output, target,
                                                             tp, fp, tn, fn)
            losses.update(loss.item(), input.size(0))
            precs.update(prec.item(), input.size(0))
            recs.update(rec.item(), input.size(0))

            # Measure elapsed time.
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {precs.val:.3f}\t'
                      'Rec {recs.val:.3f}\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       precs=precs, recs=recs))

        print(' * Loss: {loss.avg:.4f} Prec: {prec:.4f} '
              'Rec: {rec:.4f}'.format(loss=losses,
                                           prec=prec,
                                           rec=rec))

    return losses.avg, prec, rec


def save_checkpoint(state, filename, is_best=False):
    """Save torch training checkpoint."""
    create_dir_if_necessary(filename)
    torch.save(state, filename)
    if is_best:
        filename_no_ext = os.path.abspath(filename).split('.')[0]
        ext = '.'.join(filename.split('.')[1:])
        shutil.copyfile(filename, '%s_best.%s' % (filename_no_ext, ext))


def finetune(data_dir,
             checkpoint_path,
             arch='vgg16',
             dataset='voc_2007',
             ann_dir=None,
             train_split='train',
             val_split='val',
             input_size=224,
             optimizer_name='SGD',
             lr=1e-2,
             epochs=100,
             batch_size=64,
             workers=8,
             resume_checkpoint_path=None,
             should_validate=False):
    """
    Finetune the last layer of a CNN pretrained on ImageNet.

    Args:
        data_dir:
        checkpoint_path: String, path to save checkpoint.
        arch: String, name of torchvision.models architecture.
        dataset: String, name of dataset.
        input_size: Integer, length of the side of the input image.
        optimizer_name: String, name of torch.optim.Optimizer to use.
        lr: Float, learning rate to use.
        epochs: Integer, number of epochs to train for.
        batch_size: Integer, batch size to use.
        workers: Integer, number of workers to use for loading data.
        resume_checkpoint_path: String, checkpoint to resume training from.
        should_validate: Boolean, if True, validate model (no training).
    """
    # Load model, replacing the last layer.
    model = get_finetune_model(arch=arch, dataset=dataset)

    # Prepare data augmentation.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
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

        train_dset = datasets.VOCDetection(data_dir,
                                           year=year,
                                           image_set=train_split,
                                           transform=train_transform,
                                           target_transform=target_transform)
        val_dset = datasets.VOCDetection(data_dir,
                                         year=year,
                                         image_set=val_split,
                                         transform=val_transform,
                                         target_transform=target_transform)
    elif 'coco' in dataset:
        train_ann_path = os.path.join(ann_dir,
                                      'instances_%s.json' % train_split)
        val_ann_path = os.path.join(ann_dir, 'instances_%s.json' % val_split)

        target_transform = transforms.Compose([
            FromCocoToOneHotEncoding(),
            SimpleToTensor(),
        ])
        train_dset = datasets.CocoDetection(os.path.join(data_dir,
                                                         train_split),
                                            train_ann_path,
                                            transform=train_transform,
                                            target_transform=target_transform)
        val_dset = datasets.CocoDetection(os.path.join(data_dir, val_split),
                                          val_ann_path,
                                         transform=val_transform,
                                         target_transform=target_transform)
    else:
        assert(False)

    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=batch_size,
                                               num_workers=workers,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dset,
                                             batch_size=batch_size,
                                             num_workers=workers,
                                             shuffle=False,
                                             pin_memory=True)

    # Define loss criterion.
    criterion = nn.BCEWithLogitsLoss()

    # Move model to GPU or CPU.
    device = get_device()
    model = model.to(device)
    criterion = criterion.to(device)

    # Prepare optimizer.
    optimizer = torch.optim.__dict__[optimizer_name](model.parameters(), lr=lr)

    # Restore previous checkpoint, if provided.
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0

    # Validate model.
    if should_validate:
        validate(val_loader, model, criterion, device)
        return

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    best_val_loss = np.inf
    for epoch in range(start_epoch, epochs):
        curr_lr = optimizer.param_groups[0]['lr']
        if curr_lr != lr:
            print('Breaking early at %d epochs because training plateaued.'
                  % epochs)
            break
        train(train_loader, model, criterion, optimizer, epoch, device)
        val_loss, prec, rec = validate(val_loader, model, criterion, device)
        scheduler.step(val_loss)

        # Save checkpoint.
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        res = {
            'epoch': epoch + 1,
            'arch': arch,
            'dataset': dataset,
            'data_dir': data_dir,
            'input_size': input_size,
            'lr': lr,
            'prec': prec,
            'rec': rec,
            'epochs': epochs,
            'batch_size': batch_size,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(res, checkpoint_path, is_best=is_best)


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
        parser.add_argument('--ann_dir', type=str,
                            default=None,
                            help='path to root directory containing '
                                 'annotations (used for COCO)')
        parser.add_argument('--checkpoint_path', type=str,
                            default='checkpoint.pth.tar',
                            help='path to save checkpoint')
        parser.add_argument('--arch', type=str, default='vgg16',
                            help='name of CNN architecture (choose from '
                                 'PyTorch pretrained networks')
        parser.add_argument('--dataset', choices=['voc_2007', 'coco_2014'],
                            default='voc_2007',
                            help='name of dataset')
        parser.add_argument('--train_split', choices=['train', 'train2014'],
                            type=str, default='train',
                            help='name of training split')
        parser.add_argument('--val_split', choices=['val', 'val2014'],
                            type=str, default='val',
                            help='name of validation split')
        parser.add_argument('--input_size', type=int, default=224,
                            help='CNN image input size')
        parser.add_argument('--optimizer', type=str, default='SGD',
                            help='name of PyTorch optimizer')
        parser.add_argument('--lr', type=float, default=1e-2,
                            help='learning rate')
        parser.add_argument('--epochs', type=int, default=500,
                            help='number of iterations for which to train')
        parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size')
        parser.add_argument('--workers', type=int, default=8,
                            help='number of workers for data loading')
        parser.add_argument('--resume_checkpoint_path', type=str, default=None,
                            help='Path from which to resume training.')
        parser.add_argument('--validate', type=str, default=False)
        parser.add_argument('--gpu', type=int, nargs='*', default=None,
                            help='List of GPU(s) to use.')

        args = parser.parse_args()
        set_gpu(args.gpu)
        finetune(args.data_dir,
                 args.checkpoint_path,
                 arch=args.arch,
                 dataset=args.dataset,
                 ann_dir=args.ann_dir,
                 train_split=args.train_split,
                 val_split=args.val_split,
                 input_size=args.input_size,
                 optimizer_name=args.optimizer,
                 lr=args.lr,
                 epochs=args.epochs,
                 batch_size=args.batch_size,
                 workers=args.workers,
                 resume_checkpoint_path=args.resume_checkpoint_path,
                 should_validate=args.validate)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
