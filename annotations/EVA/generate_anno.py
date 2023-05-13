import os
import numpy as np
import pandas as pd
import json
from glob import glob

base_dir = '/home/dji/IAA/data/'

all_files = []
train_files = []
val_files = []
test_files = []

id_cat = {}
with open('/home/dji/IAA/data/EVA/eva-dataset-master/data/image_content_category.csv', 'r') as f:
    lines = f.readlines()

info = pd.read_csv('/home/dji/IAA/data/EVA/eva-dataset-master/data/votes_filtered.csv', sep='=', dtype={'image_id': str})
for line in lines[1:]:
    id, cat = line.strip().split(',')
    id, cat = id.strip('"'), cat.strip('"')
    score = info['score'][info['image_id']==id].to_numpy()
    if len(score) == 0:
        continue
    all_files.append({
        'image': f'EVA/eva-dataset-master/images/EVA_category/EVA_category/{cat}/{id}.jpg',
        'score': np.mean(score)
    })
    assert os.path.exists(os.path.join(base_dir, all_files[-1]['image']))




# save to json
# with open(f'{base_dir}AADB/annotations/AADB_train.json', 'w') as f:
#     json.dump({'files': train_files}, f)
# with open(f'{base_dir}AADB/annotations/AADB_val.json', 'w') as f:
#     json.dump({'files': val_files}, f)
# with open(f'{base_dir}AADB/annotations/AADB_test.json', 'w') as f:
#     json.dump({'files': test_files}, f)
with open(f'{base_dir}EVA/annotations/EVA_all.json', 'w') as f:
    json.dump({'files': all_files}, f)