import argparse
import json
import os
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='/home/dji/IAA/data/FLICKR-AES/annos.json', type=str, help='annotations json file to calculate')
parser.add_argument('--base-dir', '--base_dir', default='/home/dji/IAA/data/AADB', type=str, help='base dir of datasets')
args = parser.parse_args()

aadb = "/home/dji/IAA/data/AADB/annotations/AADB_all.json"
flickr = '/home/dji/IAA/data/FLICKR-AES/annotations/FLICKR-AES_all.json'
meta_info = {}
with open(flickr, 'r') as f:
    annos_gt = json.load(f)
for anno in annos_gt['files']:
    image, score = anno['image'], anno['score']
    img_id = image.split('/')[-1].split('_')[2]
    meta_info[img_id] = {}
    meta_info[img_id]['gt_score'] = score


with open(args.file, 'r') as f:
    annos = json.load(f)

cnt = np.zeros([101])
for year, metas in annos['files'].items():
    for meta in metas:
        favorites = meta['favorites'] + 1
        views = meta['views'] + 1
        if views <= 500:
            continue
        else:
            score = np.log(favorites) / np.log(views)
        cnt[int(score*100)] += 1

        # if score == 0:
        #     continue
        img_id = meta['rgb'].split('/')[-1].split('_')[-3]
        assert img_id in meta_info.keys()
        meta_info[img_id]['fv_score'] = score

print(sum(cnt))
print(cnt)
import matplotlib.pyplot as plt
plt.bar(range(len(cnt)), cnt)
plt.savefig('bar.png')

gt_score, fv_score = [], []
for k, scores in meta_info.items():
    if 'fv_score' not in scores.keys():
        continue
    gt_score.append(scores['gt_score'])
    fv_score.append(scores['fv_score'])

# fit
gt_score, fv_score = np.array(gt_score), np.array(fv_score)
def func(x, a, b, c, d, e):
  logist = 0.5 - 1/(1+np.exp(b * (x-c)))
  return a*logist + d*x + e
# popt, pcov = curve_fit(func, fv_score, gt_score)
# fv_score = func(fv_score, *popt)

rscc = stats.spearmanr(gt_score, fv_score)
plcc = stats.pearsonr(gt_score, fv_score)

print(rscc[0])
print(plcc[0])