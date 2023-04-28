import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='/media/dji/data3/zach_data/yfcc15m/annos_0_999999.json', type=str, help='annotations json file to calculate')
parser.add_argument('--base-dir', '--base_dir', default='/media/dji/新加卷/zach_data', type=str, help='base dir of datasets')
args = parser.parse_args()

with open(args.file, 'r') as f:
    annos = json.load(f)

cnt = np.zeros([101])
for year, metas in annos['files'].items():
    for meta in metas:
        favorites = meta['favorites'] + 1
        views = meta['views'] + 1
        if views <= 1000:
            continue
        else:
            sorce = np.log(favorites) / np.log(views)
        cnt[int(sorce*100)] += 1

print(sum(cnt))
print(cnt)
import matplotlib.pyplot as plt
plt.bar(range(len(cnt)), cnt)
plt.savefig('bar.png')
