"""
file - main.py
Main script to train the aesthetic model on the AVA dataset.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""


import os
import torch
from utils.args_parser import get_args

from nima_runner import train_nima
from tanet_runner import train_tanet


if __name__ == '__main__':
    args = get_args()

    # torch.backends.cudnn.benchmark = True
    # set GPUs to use
    if args.use_gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpus

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.model == 'nima':
        train_nima(args, device)
    if args.model == 'tanet':
        print('[EXPERIMENT BEGIN]...')
        train_tanet(args, device, False)

