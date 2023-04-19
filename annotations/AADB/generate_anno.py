import os
import numpy as np
import pandas as pd
import json
from glob import glob

base_dir = '/mnt/d/data/'

# all_files = []
train_files = []
val_files = []
test_files = []
# train set
with open(f'{base_dir}AADB/data/imgListFiles_label/imgListTrainRegression_score.txt') as f:
    for line in f:
        fs = line.strip().split(' ')
        image = f'AADB/data/datasetImages_originalSize/{fs[0]}' # image path
        score = float(fs[1]) # score
        train_files.append({'image': image, 'score': score})

# validation set
with open(f'{base_dir}AADB/data/imgListFiles_label/imgListValidationRegression_score.txt') as f:
    for line in f:
        fs = line.strip().split(' ')
        image = f'AADB/data/datasetImages_originalSize/{fs[0]}' # image path
        score = float(fs[1]) # score
        val_files.append({'image': image, 'score': score})

# test set
with open(f'{base_dir}AADB/data/imgListFiles_label/imgListTestNewRegression_score.txt') as f:
    for line in f:
        fs = line.strip().split(' ')
        image = f'AADB/data/AADB_newtest_originalSize/{fs[0]}' # image path
        score = float(fs[1]) # score
        test_files.append({'image': image, 'score': score})

# save to json
with open(f'{base_dir}AADB/annotations/AADB_train.json', 'w') as f:
    json.dump({'files': train_files}, f)
with open(f'{base_dir}AADB/annotations/AADB_val.json', 'w') as f:
    json.dump({'files': val_files}, f)
with open(f'{base_dir}AADB/annotations/AADB_test.json', 'w') as f:
    json.dump({'files': test_files}, f)
with open(f'{base_dir}AADB/annotations/AADB_all.json', 'w') as f:
    json.dump({'files': train_files + val_files + test_files}, f)