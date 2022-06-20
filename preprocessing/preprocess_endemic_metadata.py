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
train_file = csv_loader(os.path.join(root, 'train/SnakeCLEF2022-TrainMetadata.csv'))
train_endemic = np.unique([str(x['endemic']) for x in train_file])
print(len(train_file))
print(train_endemic)

endemic_label_train = {x: np.zeros(1572) for x in train_endemic}
for line in train_file:
    endemic = str(line['endemic'])
    class_id = line['class_id']
    endemic_label_train[endemic][class_id] += 1

endemic_label_train = {x: v.tolist() for x, v in endemic_label_train.items()}
json_saver(endemic_label_train, './preprocessing/endemic_label.json')
