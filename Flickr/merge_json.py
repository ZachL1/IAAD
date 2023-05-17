import os
import random
import numpy as np
import json
import argparse
from tqdm import tqdm
from PIL import Image
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', default='/home/dji/disk4/zach/iaa/data/YFCC15M/')
args = parser.parse_args()

cnt = np.zeros([101])
total = np.zeros([101])
train_files = []
test_files = []
all_score = []

def gs(x, u=22, sig=10):
    return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (np.sqrt(2*np.pi)*sig)
for s in range(101):
    total[s] = gs(s)* (50000/gs(22))
print(total)

def goog_img(file_path):
    try:
        Image.open(file_path).convert('RGB')
    except:
        print(f'[except] when read {file_path}')
        return False
    return True

def main_work(annos):
    for anno in annos:
        favs = anno['favorites'] + 1
        views = anno['views'] + 1
        new_anno = {}
        new_anno['image'] = '/home/dji/disk4/zach/iaa/data/' + 'YFCC15M/'+anno['rgb']
        new_anno['score'] = np.log(favs) / np.log(views)

        img_path = new_anno['image']
        if not os.path.exists(img_path):
            continue

        if total[int(new_anno['score']*100)] < 0:
            continue
        else:
            total[int(new_anno['score']*100)] -= 1

        if random.random() < 0.9:
            train_files.append(new_anno)
        else:
            test_files.append(new_anno)

        all_score.append(new_anno['score'])
        cnt[int(new_anno['score']*100)] += 1


with ThreadPoolExecutor(max_workers=1) as pool:
    for json_file in os.listdir(args.json_dir):
        if not json_file.endswith('filtered.json'):
            continue
        with open(os.path.join(args.json_dir, json_file), 'r') as f:
            files = json.load(f)['files']
        for year, annos in files.items():
            if int(year) < 2006 or int(year) > 2014:
                continue
            pool.submit(main_work, annos)
    

print(sum(cnt))
print(cnt)
import matplotlib.pyplot as plt
plt.bar(range(len(cnt)), cnt)
x = np.array(range(len(cnt)))
# plt.bar(x.tolist(), cnt)
y = gs(x) * (cnt[22]/gs(22))
plt.plot(x,y,'r--')

# x = np.array(all_score)
# mu = 0.22
# sigma = 0.09
# n, bins, patches = plt.hist(x, 50)
# y = norm.pdf(bins, mu, sigma)
# plt.plot(bins, y, 'r--')
plt.savefig('bar.png')


# save to json
with open(f'{args.json_dir}annotations/YFCC15M_clean_train.json', 'w') as f:
    print(f'train set: ', len(train_files))
    json.dump({'files': train_files}, f)
with open(f'{args.json_dir}annotations/YFCC15M_clean_test.json', 'w') as f:
    print(f'test set: ', len(test_files))
    json.dump({'files': test_files}, f)
with open(f'{args.json_dir}annotations/YFCC15M_clean_all.json', 'w') as f:
    json.dump({'files': train_files + test_files}, f)