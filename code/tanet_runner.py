import os
import time
from tqdm import tqdm

# import nni
# from nni.utils import merge_parameter

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models

from dataset.dataset import IAADataset
from model.TANet import TANet
from utils.loss import emd_loss
from utils.metrics import get_score, get_lcc, get_acc, get_srcc, AverageMeter, BestMeteric
from utils.logger import add_dict_scalars


def train(args, model, device, loader, optimizer, writer, epoch):
    model.train()

    # Freeze
    for name, param in model.named_parameters():
        if name[:11] == "res365_last":
            param.requires_grad = False
        else:
            param.requires_grad = True

    train_losses = AverageMeter()
    for idx, data in enumerate(tqdm(loader)):
        x = data['image'].to(device)
        y = data['ratings'].to(device).float()
        y_pred = model(x).float().view(-1,10,1)
        loss = emd_loss(y, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), x.size(0))

        print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, args.epochs, idx + 1, len(loader), loss))
        writer.add_scalar('train/emd_loss', loss, idx + epoch * len(loader))

    print('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, train_losses.avg))
    writer.add_scalar('train/epoch_emd_loss', train_losses.avg, epoch+1)
    return train_losses.avg

@torch.no_grad()
def validate(args, model, device, loader, writer=None, epoch=None, test_or_valid_flag = 'val'):
    model.eval()
    validate_losses = AverageMeter()
    torch.set_printoptions(precision=3)
    true_score = []
    pred_score = []

    for idx, data in enumerate(tqdm(loader)):
        x = data['image'].to(device)
        y = data['ratings'].to(device).float()
        y_pred = model(x).float().view(-1,10,1)
        pscore, pscore_np = get_score(y_pred, device)
        tscore, tscore_np = get_score(y, device)
        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()
        loss = emd_loss(y_pred, y).float()
        validate_losses.update(loss.item(), x.size(0))

    lcc_mean = get_lcc(pred_score, true_score)
    srcc_mean = get_srcc(pred_score, true_score)
    acc = get_acc(pred_score, true_score)
    val_result = {
        'emd_loss': validate_losses.avg,
        'acc': acc,
        'lcc': lcc_mean[0],
        'srcc': srcc_mean[0],
    }
    print(f'{test_or_valid_flag}: {val_result}')
    if writer is not None and epoch is not None:
        add_dict_scalars(writer, 'val', val_result, epoch+1)
    return val_result


def train_tanet(args, device, use_nni=True):
    ## nni 
    if use_nni:
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
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    ## init model
    model = TANet().to(device)

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
        trainset = IAADataset(annos_file=args.train_annos_file, root_dir=args.root_dir, transform=train_transform)
        valset = IAADataset(annos_file=args.val_annos_file, root_dir=args.root_dir, transform=val_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                                                   num_workers=args.num_workers, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.val_batch_size, shuffle=True, 
                                                 num_workers=args.num_workers, drop_last=True)
        
        # training loop
        while epoch < args.epochs:
            train_loss = train(args, model=model, device=device, loader=train_loader, optimizer=optimizer, writer=writer, epoch=epoch)
            val_result = validate(args, model=model, device=device, loader=val_loader, writer=writer, epoch=epoch)
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
        testset = IAADataset(annos_file=args.test_annos_file, root_dir=args.root_dir, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, 
                                                  num_workers=args.num_workers, drop_last=True)

        test_loss, tacc, tlcc, tsrcc = validate(args, model=model, device=device, loader=test_loader, test_or_valid_flag='test')
        