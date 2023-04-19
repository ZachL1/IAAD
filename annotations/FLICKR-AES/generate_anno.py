import os
import numpy as np
import pandas as pd
import json
from glob import glob

base_dir = '/mnt/d/data/'

all_files = []
train_files = []
val_files = []
test_files = []
# all set
with open(f'{base_dir}FLICKR-AES/data/FLICKR-AES_image_score.txt') as f:
    for line in f:
        fs = line.strip().split(' ')
        image = f'FLICKR-AES/data/FLICKR-AES/40K/{fs[0]}' # image path
        score = float(fs[1]) # score
        all_files.append({'image': image, 'score': score})


# save to json
# with open(f'{base_dir}AADB/annotations/AADB_train.json', 'w') as f:
#     json.dump({'files': train_files}, f)
# with open(f'{base_dir}AADB/annotations/AADB_val.json', 'w') as f:
#     json.dump({'files': val_files}, f)
# with open(f'{base_dir}AADB/annotations/AADB_test.json', 'w') as f:
#     json.dump({'files': test_files}, f)
with open(f'{base_dir}FLICKR-AES/annotations/FLICKR-AES_all.json', 'w') as f:
    json.dump({'files': all_files}, f)