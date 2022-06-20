# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

assert timm.__version__ == "0.3.2"  # version check
from timm.data.mixup import Mixup, mixup_target
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import trunc_normal_

import models_beit
import models_vit
import models_vit_get_attention
import models_vit_meta
import models_vitdet
import util.lr_decay as lrd
import util.misc as misc
from engine_finetune import evaluate, evaluate_with_attn, test, train_one_epoch
from loss.equalized_focal_loss import EqualizedFocalLoss, initial_gradient_collector
from loss.logit_adjustment_loss import LogitAdjustment, LogitAdjustmentLabelSmoothing
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import get_1d_sincos_pos_embed_from_grid, interpolate_pos_embed


class MixupReturnLam(Mixup):
    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)
        return x, target, lam


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--test_batch_size', default=None, type=int, help='Batch size per GPU during test')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument(
        '--accum_iter',
        default=1,
        type=int,
        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_file', default='models_vit', type=str, metavar='MODEL', help='Name of model file')
    parser.add_argument('--model',
                        default='vit_large_patch16',
                        type=str,
                        metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad',
                        type=float,
                        default=None,
                        metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr',
                        type=float,
                        default=1e-3,
                        metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr',
                        type=float,
                        default=1e-6,
                        metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter',
                        type=float,
                        default=None,
                        metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa',
                        type=str,
                        default='rand-m9-mstd0.5-inc1',
                        metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit',
                        action='store_true',
                        default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax',
                        type=float,
                        nargs='+',
                        default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob',
                        type=float,
                        default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob',
                        type=float,
                        default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode',
                        type=str,
                        default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--keep_head', action='store_true', default=False, help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token',
                        action='store_false',
                        dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--root', default='', type=str, help='dataset')
    parser.add_argument('--data', default='imagenet', type=str, help='dataset')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str, help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int, help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--test', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval',
                        action='store_true',
                        default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem',
                        action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--resized_crop_scale', default=[0.08, 1.0], type=float, nargs='+')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # SnakeCLEF2022
    parser.add_argument('--split', default=2, type=int)

    # metadata
    parser.add_argument('--use_meta', action='store_true', default=False)
    parser.add_argument('--meta_dims', default=[4, 3], type=int, nargs='+')

    # address prior
    parser.add_argument('--use_prior', action='store_true', default=False)

    # long-tail
    parser.add_argument('--loss', default='Base', type=str)
    parser.add_argument('--sample_per_class_file', default='./preprocessing/sample_per_class.json', type=str)

    # handle big resolution
    parser.add_argument('--mask_ratio', default=None, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_type', default='random', type=str, help='Masking type.')
    parser.add_argument('--mask_ratio_decay', action='store_true', default=False)
    parser.add_argument('--data_size', default='large', type=str)

    # test time augmentation
    parser.add_argument('--tencrop', action='store_true', default=False)
    parser.add_argument('--crop_pct', default=1.0, type=float, help='resize crop ratio')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train,
                                                            num_replicas=num_tasks,
                                                            rank=global_rank,
                                                            shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(dataset_val,
                                                              num_replicas=num_tasks,
                                                              rank=global_rank,
                                                              shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  sampler=sampler_val,
                                                  batch_size=args.test_batch_size,
                                                  num_workers=args.num_workers,
                                                  pin_memory=args.pin_mem,
                                                  drop_last=False)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = MixupReturnLam(mixup_alpha=args.mixup,
                                  cutmix_alpha=args.cutmix,
                                  cutmix_minmax=args.cutmix_minmax,
                                  prob=args.mixup_prob,
                                  switch_prob=args.mixup_switch_prob,
                                  mode=args.mixup_mode,
                                  label_smoothing=args.smoothing,
                                  num_classes=args.nb_classes)

    if args.use_meta:
        if args.model_file == 'models_vit':
            model = models_vit_meta.__dict__[args.model](
                img_size=args.input_size,
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                global_pool=args.global_pool,
                meta_dims=args.meta_dims,
            )
        elif args.model_file == 'models_vitdet':
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        if args.model_file == 'models_vit':
            model = models_vit.__dict__[args.model](
                img_size=args.input_size,
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                global_pool=args.global_pool,
                mask_ratio=args.mask_ratio,
                mask_type=args.mask_type,
            )
        elif args.model_file == 'models_vitdet':
            model = models_vitdet.__dict__[args.model](
                img_size=args.input_size,
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                global_pool=args.global_pool,
                mask_ratio=args.mask_ratio,
                mask_type=args.mask_type,
            )
        elif args.model_file == 'models_beit':
            model = models_beit.__dict__[args.model](
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                use_mean_pooling=args.global_pool,
                use_rel_pos_bias=True,
                use_abs_pos_emb=False,
                init_values=1e-6,
            )
        elif args.model_file == 'models_vit_get_attention':
            model = models_vit_get_attention.__dict__[args.model](
                img_size=args.input_size,
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                global_pool=args.global_pool,
                mask_ratio=args.mask_ratio,
                mask_type=args.mask_type,
            )
        else:
            raise NotImplementedError

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model, args)

        # load pre-trained model
        try:
            msg = model.load_state_dict(checkpoint_model, strict=False)
        except:
            print('pos_embed mismatch!')
            assert args.use_meta
            pos_embed = checkpoint['model']['pos_embed']
            meta_pos_embed = get_1d_sincos_pos_embed_from_grid(pos_embed.shape[-1],
                                                               np.arange(len(args.meta_dims), dtype=np.float32))
            meta_pos_embed = torch.from_numpy(meta_pos_embed).float().unsqueeze(0)
            print('meta_pos_embed', meta_pos_embed.shape)
            pos_embed = torch.cat((pos_embed[:, :1, :], meta_pos_embed, pos_embed[:, 1:, :]), dim=1)
            checkpoint['model']['pos_embed'] = pos_embed
            msg = model.load_state_dict(checkpoint_model, strict=False)

        # print(msg)

        if args.global_pool:
            if args.use_meta:
                for key in set(msg.missing_keys):
                    assert (key in {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}) or ('meta' in key)
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'} or set(
                    msg.missing_keys) == {'head.weight', 'head.bias'} or set(msg.missing_keys) == set()
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        if not args.keep_head:
            trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = nn.DataParallel(model).cuda()
    model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp,
                                        args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        if args.loss == 'Base':
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.loss == 'LogitAdjustment':
            criterion = LogitAdjustment(file=args.sample_per_class_file)
        elif args.loss == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        elif args.loss == 'EqualizedFocalLoss':
            criterion = EqualizedFocalLoss(num_classes=args.nb_classes)
            initial_gradient_collector(criterion, model_without_ddp.head)
        else:
            raise NotImplementedError
    elif args.smoothing > 0.:
        if args.loss == 'Base':
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        elif args.loss == 'LogitAdjustment':
            criterion = LogitAdjustmentLabelSmoothing(smoothing=args.smoothing, file=args.sample_per_class_file)
        else:
            raise NotImplementedError
    else:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.test:
        test(data_loader_val, model, device, args.output_dir, args)
        exit(0)
    if args.eval:
        EVALUATE = evaluate
        if args.model_file == 'models_vit_get_attention':
            EVALUATE = evaluate_with_attn
        test_stats = EVALUATE(data_loader_val, model, device, args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        # test_stats = evaluate(data_loader_val, model, device, args)
        # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model,
                                      criterion,
                                      data_loader_train,
                                      optimizer,
                                      device,
                                      epoch,
                                      loss_scaler,
                                      args.clip_grad,
                                      mixup_fn,
                                      log_writer=log_writer,
                                      args=args)
        if args.output_dir:
            misc.save_model(args=args,
                            model=model,
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device, args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()},
            **{f'test_{k}': v
               for k, v in test_stats.items()}, 'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
