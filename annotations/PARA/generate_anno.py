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

def get_meta(file:str):
    sub_files = []
    with open(file, 'r') as f:
        info = f.readlines()
    for line in info[1:]:
        info = line.strip().split(',')
        img_name = info[0]
        session = info[1]
        sem = info[2]
        ratings = info[3:12]
        score = info[12]
        image = f'PARA/imgs/{session}/{img_name}'
        assert os.path.exists(os.path.join(base_dir, image))
        sub_files.append({
            'image': image,
            'score': float(score),
            'ratings': [int(r) for r in ratings],
            'sem': sem,
            'session': session,
        })
    return sub_files

test_files = get_meta('/home/dji/IAA/data/PARA/annotation/PARA-GiaaTest.csv')
train_files = get_meta('/home/dji/IAA/data/PARA/annotation/PARA-GiaaTrain.csv')


# save to json
with open(f'{base_dir}PARA/annotations/PARA_train.json', 'w') as f:
    print('train_files: ', len(train_files))
    json.dump({'files': train_files}, f)

# with open(f'{base_dir}PARA/annotations/PARA_val.json', 'w') as f:
#     print('val_files: ', len(val_files))
#     json.dump({'files': val_files}, f)

with open(f'{base_dir}PARA/annotations/PARA_test.json', 'w') as f:
    print('test_files: ', len(test_files))
    json.dump({'files': test_files}, f)

with open(f'{base_dir}PARA/annotations/PARA_all.json', 'w') as f:
    json.dump({'files': train_files + test_files}, f)
