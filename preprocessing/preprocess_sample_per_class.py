import json
import os

import numpy as np
import pandas as pd


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


def json_loader(path):
    with open(path, "r") as f:
        obj = json.load(f)
    print(f'load {path}')
    return obj


def json_saver(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f'save {path}')


root = 'path/to/your/data'
file = csv_loader(os.path.join(root, 'train/SnakeCLEF2022-TrainMetadata.csv'))

sample_per_class = np.zeros(1572)

for line in file:
    gt = int(line['class_id'])
    sample_per_class[gt] += 1

json_saver(sample_per_class.tolist(), './preprocessing/sample_per_class.json')
