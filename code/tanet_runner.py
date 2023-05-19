import os
import time
from tqdm import tqdm
from functools import cmp_to_key

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models

from dataset.dataset import IAADataset
from model.TANet import TANet
from utils.loss import emd_loss
from utils.utils import sort_ckpts, save_ckpt
from utils.evaluate import evaluate, train
from utils.metrics import BestMeteric, AverageMeter


def train_tanet(args, device, use_nni=False):
    ## nni 
    if use_nni:
        import nni
        from nni.utils import merge_parameter
        tuner_params = nni.get_next_parameter()
        merge_parameter(args, tuner_params)
    
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
    model = TANet(emd=emd).to(device)

    ## dataparallel for model
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    ## init optimizer
    optimizer = optim.Adam([
        # {'params': other_params},
        {'params': model_without_ddp.res365_last.parameters(), 'lr': args.init_lr_res365_last},
        {'params': model_without_ddp.mobileNet.parameters(), 'lr': args.init_lr_mobileNet},
        {'params': model_without_ddp.head.parameters(), 'lr': args.init_lr_head},
        {'params': model_without_ddp.head_rgb.parameters(), 'lr': args.init_lr_head_rgb},
        {'params': model_without_ddp.hypernet.parameters(), 'lr': args.init_lr_hypernet},
        {'params': model_without_ddp.tygertnet.parameters(), 'lr': args.init_lr_tygertnet},
    ], lr=args.init_lr)

    ## 
    epoch = 0
    best_metrics = BestMeteric()

    ## resume model and optimizer if you wangt
    if args.resume:
        if os.path.isdir(args.resume):
            ckpt_files = sorted(os.listdir(args.resume), key=cmp_to_key(sort_ckpts))
            args.resume = os.path.join(args.resume, ckpt_files[-1])
        print('Load checkpoint: %s' % args.resume)

        # loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        # checkpoint = torch.load(args.resume, map_location=loc)
        checkpoint = torch.load(args.resume)

        model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']+1
            best_metrics = checkpoint['best_metrics']
        print('Successfully loaded model epoch-%d.pth' % epoch)

    ## Calculation parameters
    param_num = 0
    for param in model_without_ddp.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))
    # Freeze
    for name, param in model.named_parameters():
        if name[:11] == "res365_last":
            param.requires_grad = False
        else:
            param.requires_grad = True

    ## training
    if args.train:
        # Data loader
        if args.stage == 'ava':
            trainset = IAADataset(annos_file=os.path.join(args.data_dir, 'AVA/annotations/AVA_train.json'), 
                                 data_dir=args.data_dir, 
                                 ratings=True,
                                 transform=train_transform)
        elif args.stage == 'yfcc':
            trainset = IAADataset(annos_file=os.path.join(args.data_dir, 'YFCC15M/annotations/YFCC15M_train.json'), 
                                 data_dir=args.data_dir, 
                                 transform=train_transform)
        elif args.stage == 'yfcc_clean':
            trainset = IAADataset(annos_file=os.path.join(args.data_dir, 'YFCC15M/annotations/YFCC15M_clean_train.json'), 
                                 data_dir=args.data_dir, 
                                 transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   num_workers=args.num_workers, 
                                                   drop_last=True)
        
        # training loop
        while epoch < args.epochs:
            train_loss = train(args, model=model, train_loader=train_loader, device=device, optimizer=optimizer, writer=writer, epoch=epoch, emd=emd)
            val_result = evaluate(args, model, device, writer, epoch, emd)
            print(f'Epoch {epoch} completed.')

            # Use early stopping to monitor training
            if best_metrics.update(val_result):
                # save model weights if val loss decreases
                save_ckpt(args, model_without_ddp, optimizer, epoch, best_metrics)
            if best_metrics.best_times() >= args.early_stopping_patience:
                print('Val EMD loss has not decreased in %d epochs. Training terminated.' % args.early_stopping_patience)
                break

            epoch += 1

            # no need to manually update optimizer , because use nni
            # report nni
            if use_nni:
                nni.report_intermediate_result({'default': val_result['acc'], "vsrcc": val_result['srcc'], "val_loss": val_result['emd_loss']})
        
        if use_nni:
            nni.report_final_result({'default': val_result['acc'], 'vsrcc': val_result['srcc']})
        print('Training completed')
    
    elif args.test:
        test_transform = val_transform
        testset = IAADataset(annos_file=args.test_annos_file, data_dir=args.data_dir, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, 
                                                  num_workers=args.num_workers, drop_last=True)

        test_loss, tacc, tlcc, tsrcc = validate(args, model=model, device=device, loader=test_loader, test_or_valid_flag='test')
        