# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MetaFormer: https://github.com/dqshuai/MetaFormer
# --------------------------------------------------------
import json
import os

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def csv_loader(path, preprocessing=True):
    with open(path, 'r') as f:
        df = pd.read_csv(f)
    head = df.columns.values
    body = df.values
    if not preprocessing:
        return head, body
    obj = []
    for line in body:
        assert len(head) == len(line)
        obj.append({k: v for k, v in zip(head, line)})
    print(f'load {path}')
    return obj


class SnakeDataset(Dataset):
    def __init__(self, root, train, transform=None, use_meta=False, use_prior=False, test=False, data_size='large'):
        self.transform = transform
        self.use_meta = use_meta
        self.use_prior = use_prior
        self.test = test

        if train:
            self.root = os.path.join(root, f'train/SnakeCLEF2022-{data_size}_size')
            file = os.path.join(root, 'train/SnakeCLEF2022-TrainMetadata.csv')
        else:
            if not test:
                self.root = os.path.join(root, f'train/SnakeCLEF2022-{data_size}_size')
                file = os.path.join(root, 'train/SnakeCLEF2022-TrainMetadata.csv')
            else:
                self.root = os.path.join(root, 'test/SnakeCLEF2022-large_size')
                file = os.path.join(root, 'test/SnakeCLEF2022-TestMetadata.csv')

        self.samples = csv_loader(file)
        if not train and not test:
            self.samples = self.samples[:1000]

        if not test:
            self.targets = [s['class_id'] for s in self.samples]

        # handle endemic metadata
        with open('./preprocessing/endemic_label.json') as f:
            self.endemic_label_mapping = json.loads(f.read())

        # handle code metadata
        if not test:
            code_label = './preprocessing/code_label_train.json'
        else:
            code_label = './preprocessing/code_label_test.json'
        with open(code_label) as f:
            self.code_label_mapping = json.loads(f.read())

        print('num images in dataset:', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.root, sample['file_path'])

        # get target
        if not self.test:
            label = sample['class_id']
        else:
            label = -1

        # get images
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        batch = {}
        batch['images'] = img
        batch['target'] = label

        prior = torch.tensor(self.endemic_label_mapping[str(sample['endemic'])]).float()
        prior *= torch.tensor(self.code_label_mapping[sample['code']]).float()

        if self.use_prior:
            # return a 0，1 mask
            batch['prior'] = torch.ones_like(prior) * (prior > 0)

        batch['image_id'] = sample['observation_id']

        return batch
