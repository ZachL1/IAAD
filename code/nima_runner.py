import os
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models

from model.NIMA import *
from dataset.dataset import IAADataset
from utils.loss import emd_loss
from utils.evaluate import evaluate
from utils.metrics import BestMeteric

def train(args, model, train_loader, device, optimizer, writer:SummaryWriter, epoch, emd=True):
    model.train()
    
    batch_losses = []
    for i, data in enumerate(tqdm(train_loader, ncols=100, postfix=f'{epoch}/{args.epochs}epoch')):
        images = data['image'].to(device)
        if emd:
            labels = data['ratings'].to(device).float()
        else:
            labels = data['score'].to(device).float()

        outputs = model(images)
        optimizer.zero_grad()

        if emd:
            outputs = outputs.view(-1, 10, 1)
            loss = emd_loss(labels, outputs, 2)
        else:
            loss = nn.L1Loss()(labels, outputs)
        batch_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, args.epochs, i + 1, len(train_loader), loss))
        writer.add_scalar('train/emd_loss', loss, i + epoch * len(train_loader))

    avg_loss = sum(batch_losses) / len(batch_losses)
    print('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, avg_loss))
    writer.add_scalar('train/epoch_emd_loss', avg_loss, epoch+1)

    return avg_loss

def train_nima(args, device):
    ## log writer for tensorboard
    log_dir = os.path.join(args.log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    writer = SummaryWriter(log_dir=log_dir)

    ## transform
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), # resize
        transforms.RandomCrop((224, 224)), # random crop
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    ## init model
    emd = False # Whether to train with emd loss
    if args.stage == 'ava':
        emd = True
    base_model = models.vgg16(pretrained=True)
    if emd:
        model = NIMA(base_model, 10, emd).to(device)
    else:
        model = NIMA(base_model, 1, emd).to(device)

    ## dataparallel for model
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    ## init optimizer
    conv_base_lr = args.conv_base_lr
    dense_lr = args.dense_lr
    optimizer = optim.SGD([
        {'params': model_without_ddp.features.parameters(), 'lr': conv_base_lr},
        {'params': model_without_ddp.classifier.parameters(), 'lr': dense_lr}],
        lr=0, # does not matter
        momentum=0.9,
        )

    ## 
    epoch = 0
    best_metrics = BestMeteric()

    ## resume model and optimizer if you want
    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        # loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        # checkpoint = torch.load(args.resume, map_location=loc)
        checkpoint = torch.load(args.resume)

        model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            best_metrics = checkpoint['best_metrics']
        print('Successfully loaded model epoch-%d.pth' % args.resume_epoch)

    ## Calculation parameters
    param_num = 0
    for param in model_without_ddp.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))

    ## training
    if args.train:
        # Data loader
        if args.stage == 'ava':
            trainset = IAADataset(annos_file=os.path.join(args.data_dir, 'AVA/annotations/AVA_train.json'), 
                                 data_dir=args.data_dir, 
                                 ratings=True,
                                 transform=train_transform)
        elif args.stage == 'yfcc':
            trainset = IAADataset(annos_file=os.path.join(args.data_dir, 'YFCC1M/annotations/YFCC1M_train.json'), 
                                 data_dir=args.data_dir, 
                                 transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   num_workers=args.num_workers, 
                                                   drop_last=True)
        
        # training loop
        while epoch < args.epochs:
            train_loss = train(args, model, train_loader, device, optimizer, writer, epoch, emd)
            val_result = evaluate(args, model, device, writer, epoch, emd)
            
            print(f'Epoch {epoch} completed.')

            # Use early stopping to monitor training
            if best_metrics.update(val_result):
                # save model weights if val loss decreases
                save_path = os.path.join(args.ckpt_path, 'epoch-%d.pth' % (epoch + 1))
                print('Saving model to ', save_path)
                if not os.path.exists(args.ckpt_path):
                    os.makedirs(args.ckpt_path)
                torch.save({
                    'model': model_without_ddp.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_metrics': best_metrics,
                }, save_path)
                print('Done.\n')
            if best_metrics.best_times() >= args.early_stopping_patience:
                print('Val EMD loss has not decreased in %d epochs. Training terminated.' % args.early_stopping_patience)
                break

            # update optimizer: exponetial learning rate decay
            if args.decay:
                if (epoch + 1) % args.lr_decay_freq == 0:
                    conv_base_lr = conv_base_lr * args.lr_decay_rate ** ((epoch + 1) / args.lr_decay_freq)
                    dense_lr = dense_lr * args.lr_decay_rate ** ((epoch + 1) / args.lr_decay_freq)
                    optimizer = optim.SGD([
                        {'params': model_without_ddp.features.parameters(), 'lr': conv_base_lr},
                        {'params': model_without_ddp.classifier.parameters(), 'lr': dense_lr}],
                        lr=0, # does not matter
                        momentum=0.9
                    )
            
            epoch += 1

        print('Training completed.')

    if args.test:
        model.eval()
        # compute mean score
        test_transform = val_transform
        testset = IAADataset(annos_file=args.test_annos_file, data_dir=args.data_dir, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, 
                                                  num_workers=args.num_workers, drop_last=True)

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