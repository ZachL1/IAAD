import os
import numpy as np
import pandas as pd
import json
from glob import glob
import random

base_dir = '/home/dji/IAA/data/'

# all_files = []
train_files = []
# val_files = []
test_files = []

with open('/home/dji/IAA/data/TAD66K/labels/merge/test.csv', 'r') as f:
    info = f.readlines()
for line in info[1:]:
    img_name, score = line.strip().split(',')
    image = f'TAD66K/imgs/{img_name}'
    assert os.path.exists(os.path.join(base_dir, image))
    test_files.append({
        'image': image,
        'score': float(score),
    })

with open('/home/dji/IAA/data/TAD66K/labels/merge/train.csv', 'r') as f:
    info = f.readlines()
for line in info[1:]:
    img_name, score = line.strip().split(',')
    image = f'TAD66K/imgs/{img_name}'
    assert os.path.exists(os.path.join(base_dir, image))
    train_files.append({
        'image': image,
        'score': float(score),
    })



# save to json
with open(f'{base_dir}TAD66K/annotations/TAD66K_train.json', 'w') as f:
    print('train_files: ', len(train_files))
    json.dump({'files': train_files}, f)

# with open(f'{base_dir}TAD66K/annotations/TAD66K_val.json', 'w') as f:
#     print('val_files: ', len(val_files))
#     json.dump({'files': val_files}, f)

with open(f'{base_dir}TAD66K/annotations/TAD66K_test.json', 'w') as f:
    print('test_files: ', len(test_files))
    json.dump({'files': test_files}, f)

with open(f'{base_dir}TAD66K/annotations/TAD66K_all.json', 'w') as f:
    json.dump({'files': train_files + test_files}, f)
