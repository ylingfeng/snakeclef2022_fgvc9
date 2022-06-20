# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import os
import sys
import time
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import accuracy
from tqdm import tqdm

import util.lr_sched as lr_sched
import util.misc as misc
import util.utils as utils


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    mixup_fn=None,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if 'prior' in batch:
            assert args.use_prior
            prior = batch['prior'].to(device, non_blocking=True).float()
        if 'meta' in batch:
            assert args.use_meta
            meta = batch['meta'].to(device, non_blocking=True).float()
        if 'mask' in batch:
            mask = batch['mask'].to(device, non_blocking=True).float()

        images = batch['images'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)

        if mixup_fn is not None:
            images, target, lam = mixup_fn(images, target)
            if args.use_meta:
                meta = meta * lam + meta.flip(0) * (1. - lam)

        with torch.cuda.amp.autocast():
            if args.use_meta:
                output = model(images, meta)
            else:
                output = model(images)
                # output = model(images, mask)
            loss = criterion(output, target)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    y_true = []
    y_pred = []
    priors = []
    for batch in metric_logger.log_every(data_loader, 10, header):

        if 'prior' in batch:
            assert args.use_prior
            prior = batch['prior'].to(device, non_blocking=True).float()
        if 'meta' in batch:
            assert args.use_meta
            meta = batch['meta'].to(device, non_blocking=True).float()

        images = batch['images'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if args.tencrop:
                bs, ncrops, c, h, w = images.shape
                images = images.view(-1, c, h, w)
            if args.use_meta:
                output = model(images, meta)
            else:
                output = model(images)
            if args.tencrop:
                output = output.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(output, target)
            output = output.softmax(-1)

        y_true.append(target)
        y_pred.append(output)
        if args.use_prior:
            priors.append(prior)

        metric_logger.update(loss=loss.item())

    y_true = torch.cat(y_true, 0)
    y_pred = torch.cat(y_pred, 0)

    acc1, acc5 = accuracy(y_pred, y_true, topk=(1, 5))
    metric_logger.meters['acc1'].update(acc1.item(), n=len(data_loader.dataset))
    metric_logger.meters['acc5'].update(acc5.item(), n=len(data_loader.dataset))

    f1 = f1_score(y_true.cpu().numpy(), y_pred.topk(1)[1].cpu().numpy(), average='macro')
    metric_logger.meters['f1'].update(f1.item(), n=len(data_loader.dataset))

    if args.use_prior:
        priors = torch.cat(priors, 0)
        y_pred_prior = y_pred * priors

        acc1_prior, acc5_prior = accuracy(y_pred_prior, y_true, topk=(1, 5))
        metric_logger.meters['acc1_prior'].update(acc1_prior.item(), n=len(data_loader.dataset))
        metric_logger.meters['acc5_prior'].update(acc5_prior.item(), n=len(data_loader.dataset))

        f1_prior = f1_score(y_true.cpu().numpy(), y_pred_prior.topk(1)[1].cpu().numpy(), average='macro')
        metric_logger.meters['f1_prior'].update(f1_prior.item(), n=len(data_loader.dataset))

    if args.use_prior:
        utils.pickle_saver([y_true.cpu(), y_pred.cpu(), priors.cpu()], os.path.join(args.output_dir, 'val_scores.pkl'))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} F1 {f1.global_avg:.3f}'.format(
        top1=metric_logger.acc1, top5=metric_logger.acc5, f1=metric_logger.f1))
    if args.use_prior:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} F1 {f1.global_avg:.3f} use prior'.format(
            top1=metric_logger.acc1_prior, top5=metric_logger.acc5_prior, f1=metric_logger.f1_prior))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_with_attn(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    y_true = []
    y_pred = []
    y_pred_prior = []
    for batch in metric_logger.log_every(data_loader, 10, header):

        if 'prior' in batch:
            assert args.use_prior
            prior = batch['prior'].to(device, non_blocking=True).float()
        if 'meta' in batch:
            assert args.use_meta
            meta = batch['meta'].to(device, non_blocking=True).float()

        images = batch['images'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        image_id = batch['image_id']

        # compute output
        with torch.cuda.amp.autocast():
            if args.use_meta:
                output, attn = model(images, meta)
            else:
                output, attn = model(images)
            loss = criterion(output, target)
            output = output.softmax(-1)

            if args.use_prior:
                output_prior = output * prior

        ################################################################################## save images
        img = images.clone()
        B, _, H, W = images.shape
        num_heads = attn.shape[1]  # number of head
        # we keep only the output patch attention
        attn = attn[:, :, 0, 1:].reshape(B, num_heads, -1)

        L = attn.shape[-1]
        M = int(L**0.5)

        attn = attn.reshape(B, num_heads, M, M)
        attn = F.interpolate(attn, (H, W), mode="bilinear")  # B, num_heads, H, W

        mean = np.array(IMAGENET_DEFAULT_MEAN).reshape(1, 1, -1)
        std = np.array(IMAGENET_DEFAULT_STD).reshape(1, 1, -1)

        img = img.permute(0, 2, 3, 1).cpu().numpy()
        # print(attn.max(), attn.min())

        correct = (target == output_prior.topk(k=1)[1].reshape(-1))

        # attn = attn.permute(0, 2, 3, 1).cpu().numpy()
        # print(attn.max(), attn.min())
        # for i in range(B):
        #     real_img = cv2.cvtColor(np.uint8(255 * ((img[i] * std) + mean)), cv2.COLOR_RGB2BGR)
        #     for j in range(num_heads):
        #         fname = f'images/{img[i][:2,0,0]}_{j}.png'
        #         plt.imsave(fname=fname, arr=attn[i, :, :, j], format='png')
        #         attn_img = cv2.imread(fname)
        #         cv2.imwrite(fname, np.concatenate([real_img, attn_img], 1))

        attn = attn.permute(0, 2, 3, 1).cpu().mean(-1).numpy()
        for i in range(B):
            fname = f'images/{image_id[i]}_mean_{correct[i]}.png'
            real_img = cv2.cvtColor(np.uint8(255 * ((img[i] * std) + mean)), cv2.COLOR_RGB2BGR)
            plt.imsave(fname=fname, arr=attn[i], format='png')
            attn_img = cv2.imread(fname)
            cv2.imwrite(fname, np.concatenate([real_img, attn_img], 1))
        ##################################################################################

        y_true.append(target)
        y_pred.append(output.topk(k=1)[1])
        y_pred_prior.append(output_prior.topk(k=1)[1])
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_prior, acc5_prior = accuracy(output_prior, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['acc1_prior'].update(acc1_prior.item(), n=batch_size)
        metric_logger.meters['acc5_prior'].update(acc5_prior.item(), n=batch_size)

    y_true = torch.cat(y_true, 0)
    y_pred = torch.cat(y_pred, 0)
    y_pred_prior = torch.cat(y_pred_prior, 0)
    f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
    f1_prior = f1_score(y_true.cpu().numpy(), y_pred_prior.cpu().numpy(), average='macro')
    metric_logger.meters['f1'].update(f1.item(), n=len(data_loader.dataset))
    metric_logger.meters['f1_prior'].update(f1_prior.item(), n=len(data_loader.dataset))

    # utils.pickle_saver([y_true.cpu().numpy(), y_pred_prior.cpu().numpy()], '/data/jupyter/longtail.pkl')
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} F1 {f1.global_avg:.3f}'.format(
        top1=metric_logger.acc1, top5=metric_logger.acc5, f1=metric_logger.f1))
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} F1 {f1.global_avg:.3f} use prior'.format(
        top1=metric_logger.acc1_prior, top5=metric_logger.acc5_prior, f1=metric_logger.f1_prior))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test(data_loader, model, device, output_dir, args):
    # switch to evaluation mode
    model.eval()

    y_pred = []
    priors = []
    image_ids = []
    for i, batch in enumerate(tqdm(data_loader, ncols=100)):

        if 'prior' in batch:
            assert args.use_prior
            prior = batch['prior'].to(device, non_blocking=True).float()
        if 'meta' in batch:
            assert args.use_meta
            meta = batch['meta'].to(device, non_blocking=True).float()

        images = batch['images'].to(device, non_blocking=True)
        image_id = batch['image_id']

        # compute output
        with torch.cuda.amp.autocast():
            if args.tencrop:
                bs, ncrops, c, h, w = images.shape
                images = images.view(-1, c, h, w)
            if args.use_meta:
                output = model(images, meta)
            else:
                output = model(images)
            if args.tencrop:
                output = output.view(bs, ncrops, -1).mean(1)  # avg over crops
            output = output.softmax(-1)

            y_pred.append(output)
            if args.use_prior:
                priors.append(prior)
            image_ids.append(image_id)
        # if i == 2: break

    y_pred = torch.cat(y_pred, 0)
    if args.use_prior:
        priors = torch.cat(priors, 0)
        y_pred_prior = y_pred * priors
    image_ids = torch.cat(image_ids, 0)

    if args.use_prior:
        utils.pickle_saver(
            [image_ids, y_pred.cpu(), priors.cpu()],
            os.path.join(
                args.output_dir,
                f'tencrop_{args.tencrop}_crop_pct_{args.crop_pct}_epoch_{args.start_epoch - 1}_test_scores.pkl'))

    results = {}
    for image_id, output in zip(image_ids, y_pred_prior):
        # output: torch.tensor(num_classes)
        image_id = int(image_id)
        if image_id in results:
            results[image_id].append(output)
        else:
            results[image_id] = [output]

    image_ids = [k for k, v in results.items()]
    pred_ids = [int(torch.stack(v, 0).mean(0).topk(k=1)[1]) for k, v in results.items()]

    dataframe = pd.DataFrame({'ObservationId': image_ids, 'class_id': pred_ids})

    creat_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    dataframe.to_csv(os.path.join(output_dir, '%s_test.csv' % creat_time), index=False, sep=',')
    print('save', os.path.join(output_dir, '%s_test.csv' % creat_time))
    return y_pred