import os
import random
import numpy as np
import json
import argparse
from tqdm import tqdm
from PIL import Image
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', default='/home/dji/disk4/zach/iaa/data/YFCC15M/')
args = parser.parse_args()

cnt = np.zeros([101])
total = np.zeros([101])
train_files = []
test_files = []
all_score = []


with open(os.path.join(args.json_dir, 'annotations/YFCC15M_all.json'), 'r') as f:
    annos = json.load(f)['files']

for anno in annos:
    if anno['score'] == 0:
        continue

    if random.random() < 0.9:
        train_files.append(anno)
    else:
        test_files.append(anno)

    all_score.append(anno['score'])

# save to json
with open(f'{args.json_dir}annotations/YFCC15M_clean_train.json', 'w') as f:
    print(f'train set: ', len(train_files))
    json.dump({'files': train_files}, f)
with open(f'{args.json_dir}annotations/YFCC15M_clean_test.json', 'w') as f:
    print(f'test set: ', len(test_files))
    json.dump({'files': test_files}, f)
with open(f'{args.json_dir}annotations/YFCC15M_clean_all.json', 'w') as f:
    json.dump({'files': train_files + test_files}, f)


