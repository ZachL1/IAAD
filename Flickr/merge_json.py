import os
import random
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', default='/home/dji/IAA/data/YFCC1M/annotations/')
args = parser.parse_args()

cnt = np.zeros([101])
train_files = []
test_files = []
for json_file in os.listdir(args.json_dir):
    if not json_file.endswith('filtered.json'):
        continue
    with open(os.path.join(args.json_dir, json_file), 'r') as f:
        files = json.load(f)['files']
    for year, annos in files.items():
        if int(year) < 2006 or int(year) > 2014:
            continue
        for anno in annos:
            favs = anno['favorites'] + 1
            views = anno['views'] + 1
            new_anno = {}
            new_anno['image'] = 'YFCC1M/'+anno['rgb']
            new_anno['score'] = np.log(favs) / np.log(views)
            if random.random() < 0.9:
                train_files.append(new_anno)
            else:
                test_files.append(new_anno)

            cnt[int(new_anno['score']*100)] += 1

print(sum(cnt))
print(cnt)
import matplotlib.pyplot as plt
plt.bar(range(len(cnt)), cnt)
plt.savefig('bar.png')

# save to json
with open(f'{args.json_dir}YFCC1M_train.json', 'w') as f:
    print(f'train set: ', len(train_files))
    json.dump({'files': train_files}, f)
with open(f'{args.json_dir}YFCC1M_test.json', 'w') as f:
    print(f'test set: ', len(test_files))
    json.dump({'files': test_files}, f)
with open(f'{args.json_dir}YFCC1M_all.json', 'w') as f:
    json.dump({'files': train_files + test_files}, f)