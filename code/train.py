"""
file - main.py
Main script to train the aesthetic model on the AVA dataset.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""


import os

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models


from utils.args_parser import get_args
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
from dataset.dataset import IAADataset

# for NIMA
from model.NIMA import *


def train_nima(config, device):
    writer = SummaryWriter()

    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model)

    # distribute model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model.to(device),
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank)
    #     model_without_ddp = model.module
    # else:
    #     if torch.cuda.device_count() > 1:
    #         # print('Use %d GPUs' % torch.cuda.device_count())
    #         print('Use GPUs: ', config.gpu_ids)
    #         model = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
    #         model_without_ddp = model.module

    #         model = model.to(device)
    #     else:
    #         model_without_ddp = model

    if config.resume:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-%d.pth' % config.resume_epoch)))
        print('Successfully loaded model epoch-%d.pth' % config.resume_epoch)

    if config.multi_gpu:
        print(config.gpu_ids)
        model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
        )

    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))

    if config.train:
        trainset = IAADataset(annos_file=config.train_annos_file, root_dir=config.root_dir, transform=train_transform)
        valset = IAADataset(annos_file=config.val_annos_file, root_dir=config.root_dir, transform=val_transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
            shuffle=True, num_workers=config.num_workers)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
            shuffle=False, num_workers=config.num_workers)
        # for early stopping
        count = 0
        init_val_loss = float('inf')
        train_losses = []
        val_losses = []
        for epoch in range(config.resume_epoch, config.epochs):
            model.train()
            
            batch_losses = []
            for i, data in enumerate(train_loader):
                images = data['image'].to(device)
                labels = data['ratings'].to(device).float()
                outputs = model(images)
                outputs = outputs.view(-1, 10, 1)

                optimizer.zero_grad()

                loss = emd_loss(labels, outputs)
                batch_losses.append(loss.item())

                loss.backward()

                optimizer.step()

                print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss.data[0]))
                writer.add_scalar('batch train loss', loss.data[0], i + epoch * (len(trainset) // config.train_batch_size + 1))

            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, avg_loss))

            # exponetial learning rate decay
            if config.decay:
                if (epoch + 1) % 10 == 0:
                    conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                    dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                    optimizer = optim.SGD([
                        {'params': model.features.parameters(), 'lr': conv_base_lr},
                        {'params': model.classifier.parameters(), 'lr': dense_lr}],
                        momentum=0.9
                    )

            # do validation after each epoch
            batch_val_losses = []
            for data in val_loader:
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                with torch.no_grad():
                    outputs = model(images)
                outputs = outputs.view(-1, 10, 1)
                val_loss = emd_loss(labels, outputs)
                batch_val_losses.append(val_loss.item())
            avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
            val_losses.append(avg_val_loss)
            print('Epoch %d completed. Mean EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))
            writer.add_scalars('epoch losses', {'epoch train loss': avg_loss, 'epoch val loss': avg_val_loss}, epoch + 1)

            # Use early stopping to monitor training
            if avg_val_loss < init_val_loss:
                init_val_loss = avg_val_loss
                # save model weights if val loss decreases
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)
                torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'epoch-%d.pth' % (epoch + 1)))
                print('Done.\n')
                # reset count
                count = 0
            elif avg_val_loss >= init_val_loss:
                count += 1
                if count == config.early_stopping_patience:
                    print('Val EMD loss has not decreased in %d epochs. Training terminated.' % config.early_stopping_patience)
                    break

        print('Training completed.')

        '''
        # use tensorboard to log statistics instead
        if config.save_fig:
            # plot train and val loss
            epochs = range(1, epoch + 2)
            plt.plot(epochs, train_losses, 'b-', label='train loss')
            plt.plot(epochs, val_losses, 'g-', label='val loss')
            plt.title('EMD loss')
            plt.legend()
            plt.savefig('./loss.png')
        '''

    if config.test:
        model.eval()
        # compute mean score
        test_transform = val_transform
        testset = IAADataset(annos_file=config.test_annos_file, root_dir=config.root_dir, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)

        mean_preds = []
        std_preds = []
        for data in test_loader:
            image = data['image'].to(device)
            output = model(image)
            output = output.view(10, 1)
            predicted_mean, predicted_std = 0.0, 0.0
            for i, elem in enumerate(output, 1):
                predicted_mean += i * elem
            for j, elem in enumerate(output, 1):
                predicted_std += elem * (j - predicted_mean) ** 2
            predicted_std = predicted_std ** 0.5
            mean_preds.append(predicted_mean)
            std_preds.append(predicted_std)
        # Do what you want with predicted and std...


if __name__ == '__main__':
    args = get_args()

    # torch.backends.cudnn.benchmark = True

    # distributed setting
    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True

        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))

        setup_for_distributed(args.local_rank == 0)

    if args.model == 'nima':
        train_nima(args, device)

