import os
from functools import cmp_to_key

def sort_ckpts(ckptl:str, ckptr:str):
    epochl = int(ckptl.split('-')[1][:-4])
    epochr = int(ckptr.split('-')[1][:-4])
    return epochl - epochr

basedir = './ckpts/'
for exp in os.listdir(basedir):
    exp_dir = os.path.join(basedir, exp)
    ckpt_files = sorted(os.listdir(exp_dir), key=cmp_to_key(sort_ckpts))
    for ckpt_file in ckpt_files[:-10]:
        cmd = f'rm -f {os.path.join(exp_dir, ckpt_file)}'
        os.system(cmd)
     