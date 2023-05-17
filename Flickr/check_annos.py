import argparse
import pickle
import json
import os
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='annos_all.json', type=str, help='annotations json file to check')
parser.add_argument('--base-dir', '--base_dir', default='/home/dji/disk4/zach/iaa/data/', type=str, help='base dir of datasets')
args = parser.parse_args()


def check_file(file:str):
    file = os.path.join(args.base_dir, file)
    if not os.path.exists(file):
        print(f'[ERROR]: not found {file}')
        return False
    # print(f'{file}\t OK')
    return True

def check_dict(d:dict):
    for file in d.values():
        check_file(file)

def check_pick(pick_file:str):
    if not check_file(pick_file):
        return
    with open(os.path.join(args.base_dir, pick_file), 'rb') as f:
        pick = pickle.load(f)
    for k, v in pick.items():
        if isinstance(v, dict):
            for d in v.values():
                if isinstance(d, dict):
                    check_dict(d)
                else:
                    check_file(d)
        elif isinstance(v, str):
            check_file(v)
        


with open(args.file, 'r') as f:
    annos = json.load(f)

# check rgb
for anno in annos['files']:
    check_file(anno['image'])


# with ThreadPoolExecutor(max_workers=20) as pool:
#     for anno in annos['files']:
#         for pick in anno.values():
#             if not pick is None:
#                 # check_pick(pick)
#                 pool.submit(check_pick, pick)