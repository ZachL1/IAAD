import os
import time
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models

from model.NIMA import *
from dataset.dataset import IAADataset
from utils.loss import emd_loss
from utils.metrics import get_score, get_lcc, get_acc, get_srcc, BestMeteric
from utils.logger import add_dict_scalars

def train(args, model, train_loader, device, optimizer, writer:SummaryWriter, epoch, ):
    model.train()
    
    batch_losses = []
    for i, data in enumerate(tqdm(train_loader)):
        images = data['image'].to(device)
        labels = data['ratings'].to(device).float()
        outputs = model(images)
        outputs = outputs.view(-1, 10, 1)

        optimizer.zero_grad()

        loss = emd_loss(labels, outputs, 2)
        batch_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, args.epochs, i + 1, len(train_loader), loss))
        writer.add_scalar('train/emd_loss', loss, i + epoch * len(train_loader))

    avg_loss = sum(batch_losses) / len(batch_losses)
    print('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, avg_loss))
    writer.add_scalar('train/epoch_emd_loss', avg_loss, epoch+1)

    return avg_loss

@torch.no_grad()
def validate(args, model, val_loader, device, writer=None, epoch=None, test_or_valid_flag = 'val'):
    model.eval()

    batch_val_losses = []
    true_score = []
    pred_score = []
    for data in tqdm(val_loader):
        images = data['image'].to(device)
        labels = data['ratings'].to(device).float()
        outputs = model(images)
        outputs = outputs.view(-1, 10, 1)
        # score for eval metrics
        pscore, pscore_np = get_score(outputs, device)
        tscore, tscore_np = get_score(labels, device)
        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()
        # emd loss
        val_loss = emd_loss(labels, outputs)
        # val_loss = emd_loss(labels, outputs, 1)
        batch_val_losses.append(val_loss.item())

    lcc_mean = get_lcc(pred_score, true_score)
    srcc_mean = get_srcc(pred_score, true_score)
    acc = get_acc(pred_score, true_score)
    avg_val_loss = sum(batch_val_losses) / len(batch_val_losses)
    val_result = {
        'emd_loss': avg_val_loss,
        'acc': acc,
        'lcc': lcc_mean[0],
        'srcc': srcc_mean[0],
    }
    print(f'{test_or_valid_flag}: {val_result}')
    if writer is not None and epoch is not None:
        add_dict_scalars(writer, 'val', val_result, epoch+1)
    return val_result


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
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    ## init model
    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model).to(device)

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
        trainset = IAADataset(annos_file=args.train_annos_file, root_dir=args.root_dir, transform=train_transform)
        valset = IAADataset(annos_file=args.val_annos_file, root_dir=args.root_dir, transform=val_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                                                   num_workers=args.num_workers, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.val_batch_size, shuffle=False, 
                                                 num_workers=args.num_workers, drop_last=True)
        
        # training loop
        while epoch < args.epochs:
            train_loss = train(args, model, train_loader, device, optimizer, writer, epoch)
            val_result = validate(args, model, val_loader, device, writer, epoch)
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
        testset = IAADataset(annos_file=args.test_annos_file, root_dir=args.root_dir, transform=test_transform)
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