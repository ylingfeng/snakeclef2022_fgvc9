import os
import time

import pandas as pd
import torch

from util.utils import pickle_loader


def gen_csv(y_pred, image_ids, output_dir, merge_type='mean'):
    results = {}
    for image_id, output in zip(image_ids, y_pred):
        # output: torch.tensor(num_classes)
        image_id = int(image_id)
        if image_id in results:
            results[image_id].append(output)
        else:
            results[image_id] = [output]

    image_ids = [k for k, v in results.items()]
    if merge_type == 'mean':
        pred_ids = [int(torch.stack(v, 0).mean(0).topk(k=1)[1]) for k, v in results.items()]
    elif merge_type == 'max':
        pred_ids = [int(torch.stack(v, 0).max(0)[0].topk(k=1)[1]) for k, v in results.items()]

    dataframe = pd.DataFrame({'ObservationId': image_ids, 'class_id': pred_ids})

    creat_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    dataframe.to_csv(os.path.join(output_dir, '%s_test.csv' % creat_time), index=False, sep=',')
    print('save', os.path.join(output_dir, '%s_test.csv' % creat_time))


output_list = [
    ['./output_dir/vit_huge_patch14_392_40e/tencrop_True_crop_pct_0.875_test_scores.pkl', 3],
    ['./output_dir/vit_large_patch16_432_50e/tencrop_True_crop_pct_0.875_test_scores.pkl', 1],
]

ensemble_scores = []

global_prior = None

for line in output_list:
    output_dir, weight = line
    # before merge
    print(weight)
    val_scores = pickle_loader(output_dir)
    image_ids, y_pred, priors = val_scores
    if global_prior is None:
        global_prior = priors
    assert (global_prior != priors).sum() == 0
    assert y_pred.max() <= 1

    # print(image_ids.shape, y_pred.shape, priors.shape)
    y_pred_prior = y_pred * global_prior
    ensemble_scores.append(y_pred_prior * weight)

ensemble_scores = torch.stack(ensemble_scores).mean(0)

# refine_dict = {x['test_id']: x for x in sorted_id_to_label[:249]}

gen_csv(ensemble_scores, image_ids, './ensemble')
