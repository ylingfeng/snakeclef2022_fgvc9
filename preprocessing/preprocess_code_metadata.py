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
#----------------------------------------------------------------#
# train
train_file = csv_loader(os.path.join(root, 'train/SnakeCLEF2022-TrainMetadata.csv'))
train_code = np.unique([x['code'] for x in train_file])
print(len(train_file))
print(len(train_code))

code_label_train = {x: np.zeros(1572) for x in train_code}
for line in train_file:
    code = line['code']
    class_id = line['class_id']
    code_label_train[code][class_id] += 1

assert 'unknown' in code_label_train
code_label_train['unknown'] = np.ones(1572)

code_label_train = {x: v.tolist() for x, v in code_label_train.items()}
json_saver(code_label_train, './preprocessing/code_label_train.json')

#----------------------------------------------------------------#
# test
test_file = csv_loader(os.path.join(root, 'test/SnakeCLEF2022-TestMetadata.csv'))
test_code = np.unique([x['code'] for x in test_file])
print(len(test_file))
print(len(test_code))

ood = []
code_label_test = {}
for code in test_code:
    if code in code_label_train and code != 'unknown':
        code_label_test[code] = code_label_train[code]
    else:
        ood.append(code)
        code_label_test[code] = np.ones(1572).tolist()

print(ood)

class2id = {}
for line in train_file:
    cname = line['binomial_name']
    cid = line['class_id']
    if cname not in class2id:
        class2id[cname] = cid
    else:
        assert class2id[cname] == cid
address_label_ISO = {}
head, file = csv_loader(os.path.join(root, 'SnakeCLEF2022-ISOxSpeciesMapping.csv'), False)
print(len(file[:, 0]))
ids = [class2id[fn] for fn in file[:, 0]]
assert ids == list(range(1572))

for i, h in enumerate(head):
    if i == 0: continue
    address_label_ISO[h] = file[:, i].tolist()

print(len(address_label_ISO))

name_code_file = json_loader('./preprocessing/slim-2.json')
name_code_mapping = {}
for line in name_code_file:
    #print(line)
    name = line['name'].lower()
    code = line['alpha-2']
    name_code_mapping[name] = code
name_code_mapping['bolivia, plurinational state of'] = 'BO'
name_code_mapping['congo, the democratic republic of the'] = 'CD'
name_code_mapping['iran, islamic republic of'] = 'IR'
name_code_mapping["côte d' ivoire"] = 'CI'
name_code_mapping['united kingdom'] = 'GB'
name_code_mapping['united states'] = 'US'
name_code_mapping['venezuela, bolivarian republic of'] = 'VE'
name_code_mapping['curacao'] = 'CW'
name_code_mapping['sint eustatius and saba'] = 'BQ'
name_code_mapping['bonaire'] = 'BQ'
name_code_mapping['british virgin islands'] = 'VG'
name_code_mapping['us virgin islands'] = 'VI'
name_code_mapping['saint barthelemy'] = 'BL'
name_code_mapping['aland islands'] = 'AX'
name_code_mapping['reunion'] = 'RE'
name_code_mapping['north korea'] = name_code_mapping["korea (democratic people's republic of)"]
name_code_mapping['laos'] = name_code_mapping["lao people's democratic republic"]
name_code_mapping['vatican'] = 'VA'
name_code_mapping['kosovo'] = 'XK'

code_name_mapping = {v: k for k, v in name_code_mapping.items()}

code_label_ISO = {}
for name, label in address_label_ISO.items():
    label = np.array(label)
    name = name.replace('.', ' ').lower()

    if name in name_code_mapping:
        code = name_code_mapping[name]
    else:
        print(f'skip {name}')
        continue

    if code in code_label_ISO:
        print('duplicate', code)
        code_label_ISO[code] += label
    else:
        code_label_ISO[code] = label

new_code_label_test = {}
for k, v in code_label_test.items():
    if k in code_label_ISO and sum(code_label_ISO[k]) != 0:
        a = np.array(code_label_ISO[k]) > 0  # mapping
        b = np.array(code_label_train[k]) > 0  # 统计量
        c = np.array(code_label_test[k]) > 0  # 统计量
        assert (b != c).sum() == 0

        new_code_label_test[k] = (a | b).astype(float).tolist()
        #print(a.sum(), b.sum(), sum(new_code_label_test[k]))
    else:
        new_code_label_test[k] = code_label_test[k]
        print(k, (np.array(new_code_label_test[k]) > 0).sum())
json_saver(new_code_label_test, './preprocessing/code_label_test.json')
