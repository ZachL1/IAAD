import argparse
import pickle
import json
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='IAAD/Flickr/yfcc2m_annos_all.json', type=str, help='annotations json file to calculate')
parser.add_argument('--base-dir', '--base_dir', default='/media/dji/新加卷/zach_data', type=str, help='base dir of datasets')
args = parser.parse_args()

base_len = len(args.base_dir) + 1
def abs2rela(abs_path: str):
    return abs_path[base_len:]

def move_to(old:str, new:str):
    new_dir = os.path.dirname(new)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    os.rename(os.path.join(args.base_dir, old), os.path.join(args.base_dir, new))

cnt = np.zeros([101])
files = {}

def pick_handler(year:str, picks:list):
    # sub_cnt = np.zeros([101])
    for pick in tqdm(picks):
        pick_path = os.path.join(args.base_dir, pick)
        new_pick_path = pick_path
        
        if not os.path.exists(pick_path):
            pick_path = glob(pick_path.replace('.pkl', '*.pkl'))[0]

        new_rgb_path = rgb_path = None
        with open(pick_path, 'rb+') as f:
            meta_info = pickle.load(f)
            post_year = time.ctime(int(meta_info['dates']['posted']))[-4:]
            if year != post_year:
                new_pick_path = pick_path.replace(f'/{year}/', f'/{post_year}/')
                rgb_path = meta_info['rgb']
                new_rgb_path = rgb_path.replace(f'/{year}/', f'/{post_year}/')
                meta_info['rgb'] = new_rgb_path
                pickle.dump(meta_info, f)
        
        favorites = int(meta_info['favorites']) + 1
        views = int(meta_info['views']) + 1
        if views <= 1:
            continue
        else:
            sorce = np.log(favorites)/np.log(views)
        cnt[int(sorce*100)] += 1

        if rgb_path is not None and new_rgb_path is not None:
            move_to(rgb_path, new_rgb_path)

        new_pick_path = new_pick_path.replace('.pkl', f'_{meta_info["favorites"]}_{meta_info["views"]}.pkl')
        move_to(pick_path, new_pick_path)
        if post_year not in files.keys():
            files[post_year] = []
        files[post_year].append(abs2rela(new_pick_path))
        
    # global cnt
    # cnt += sub_cnt
    print(f'{year} year done!')



with open(args.file, 'r') as f:
    annos = json.load(f)

all_picks = []
# with ThreadPoolExecutor(max_workers=1) as pool:
for year, picks in annos['files'].items():
    pick_handler(year, picks)
        # all_picks = all_picks + picks
        # pool.submit(pick_handler, year, picks)

# pick_handler('2000', all_picks)
print(cnt)
import matplotlib.pyplot as plt
plt.bar(range(len(cnt)), cnt)
plt.savefig('bar.png')

with open('new_yfcc2m_annos_all.json', 'w') as f:
    json.dump({'files': files}, f)