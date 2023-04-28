import os
import numpy as np
import pandas as pd
import json
from glob import glob
import random

base_dir = '/home/dji/IAA/data/'

# all_files = []
train_files = []
val_files = []
test_files = []

def get_mean(ratings:list):
    sum_rat = 0
    for rat in range(1,11):
        sum_rat += ratings[rat-1] * rat
    return sum_rat / sum(ratings)

def read_tag_challenges(file_path:str) -> dict:
    tags = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            spl = line.find(' ')
            tag_id, tag = line[:spl], line[spl+1:]
            tags[int(tag_id)] = tag
    return tags

# read tags and challenges
tags = read_tag_challenges(os.path.join(base_dir, 'AVA/data/tags.txt'))
challenges = read_tag_challenges(os.path.join(base_dir, 'AVA/data/challenges.txt'))


# read meta_data
meta_data = pd.read_csv(os.path.join(base_dir, 'AVA/data/AVA.txt'), sep=' ', index_col=1, header=None, dtype={1:str})

# generate json by categories
train_sets = glob(os.path.join(base_dir, 'AVA/data/aesthetics_image_lists/*_train.jpgl'))
test_sets = glob(os.path.join(base_dir, 'AVA/data/aesthetics_image_lists/*_test.jpgl'))

def get_json_from_ids(id_set:list):
    files = []
    for id in id_set:
        if id not in meta_data.index:
            continue
        image_file = os.path.join(base_dir, f'AVA/data/image/{id}.jpg')
        if not os.path.exists(image_file):
            continue

        meta = meta_data.loc[id,:].to_numpy().tolist()
        ratings = meta[1:11]

        sem = []
        for sem_tag in meta[11:13]:
            if sem_tag != 0:
                sem.append(tags[sem_tag])

        sub_meta = {
            'image': f'AVA/data/image/{id}.jpg', 
            'score': get_mean(ratings),
            'ratings': ratings,
            'sem': sem,
            'challenge': challenges[meta[-1]],
        }
        files.append(sub_meta)
    return files

# def get_json_from_set(sets:list):
#     files = []
#     for set in sets:
#         id_set = []
#         with open(set, 'r') as f:
#             for line in f:
#                 id_set.append(line.strip())

#         set_name = os.path.basename(set)
#         categorie = set_name[:set_name.rfind('_')]

#     return files

# train_files = get_json_from_set(train_sets)
# test_files = get_json_from_set(test_sets)

all_files = get_json_from_ids(meta_data.index.to_list())
random.shuffle(all_files)
l = len(all_files)
train_files = all_files[:int(l*0.8)]
val_files = all_files[int(l*0.8):int(l*0.85)]
test_files = all_files[int(l*0.85):]

# save to json
with open(f'{base_dir}AVA/annotations/AVA_train.json', 'w') as f:
    print('train_files: ', len(train_files))
    json.dump({'files': train_files}, f)

with open(f'{base_dir}AVA/annotations/AVA_val.json', 'w') as f:
    print('val_files: ', len(val_files))
    json.dump({'files': val_files}, f)

with open(f'{base_dir}AVA/annotations/AVA_test.json', 'w') as f:
    print('test_files: ', len(test_files))
    json.dump({'files': test_files}, f)

with open(f'{base_dir}AVA/annotations/AVA_all.json', 'w') as f:
    json.dump({'files': all_files}, f)
