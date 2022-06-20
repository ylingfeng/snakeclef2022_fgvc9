# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import os

import dataset_nori
import PIL
import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data == 'imagenet':
        assert args.nb_classes == 1000
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.data == 'snakeclef2022':
        assert args.nb_classes == 1572
        dataset = dataset_nori.SnakeDataset(args.root,
                                            is_train,
                                            transform=transform,
                                            use_meta=args.use_meta,
                                            use_prior=args.use_prior,
                                            test=args.test,
                                            data_size=args.data_size)
    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            scale=args.resized_crop_scale,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        # crop_pct = 1.0
        crop_pct = args.crop_pct
    print('crop_pct:', crop_pct)

    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    if args.tencrop:
        t.append(transforms.TenCrop(args.input_size))
        t.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        t.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean, std)(crop) for crop in crops])))
    else:
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
