import json
import pickle

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


def csv_saver(obj, path):
    dataframe = pd.DataFrame(obj)
    with open(path, "w") as f:
        dataframe.to_csv(f, index=False, sep=',')
    print(f'save {path}')


def json_loader(path):
    with open(path, "r") as f:
        obj = json.load(f)
    print(f'load {path}')
    return obj


def json_saver(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f'save {path}')


def txt_loader(path):
    with open(path, "r") as f:
        obj = f.read().splitlines()
    print(f'load {path}')
    return obj


def txt_saver(obj, path):
    print(f'save {path}')


def pickle_loader(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print(f'load {path}')
    return obj


def pickle_saver(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    print(f'save {path}')
