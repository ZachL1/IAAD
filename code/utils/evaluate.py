import os
from tqdm import tqdm
import numpy as np

import torch
import torchvision.transforms as transforms

from dataset.dataset import IAADataset
from utils.loss import emd_loss
from utils.metrics import get_score, get_lcc, get_acc, get_srcc, get_l1, get_l2, score_mapping, BestMeteric
from utils.logger import add_dict_scalars

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])])

@torch.no_grad()
def validate(args, model, device, val_loader, writer=None, epoch=None, test_or_valid_flag='val', emd=True):
    model.eval()
    torch.set_printoptions(precision=3)

    batch_val_losses = []
    true_score = []
    pred_score = []
    for data in tqdm(val_loader, ncols=100, postfix=f'{epoch}/{args.epochs}epoch'):
        images = data['image'].to(device)
        outputs = model(images)

        # emd loss train, and eval data has ratings
        if emd and 'ratings' in data.keys():
            outputs = outputs.view(-1, 10, 1)
            labels = data['ratings'].to(device).float()
            # emd loss
            val_loss = emd_loss(labels, outputs)
            # score for eval metrics
            outputs = get_score(outputs, device)
            labels = get_score(labels, device)
            batch_val_losses.append(val_loss.item())
        # emd loss train, but eval data has no ratings
        elif emd:
            outputs = outputs.view(-1, 10, 1)
            labels = data['score']
            outputs = get_score(outputs, device)
        # L1 loss train
        else:
            labels = data['score']
        outputs = outputs.view(-1,1)
        labels = labels.view(-1,1)
        pred_score += outputs.tolist()
        true_score += labels.tolist()
        
    pred_score, true_score = np.array(pred_score).reshape(-1), np.array(true_score).reshape(-1)
    val_result = {}
    if len(batch_val_losses) > 0:
        avg_val_loss = sum(batch_val_losses) / len(batch_val_losses)
        val_result = {
            'emd_loss': avg_val_loss,
            'acc': get_acc(pred_score, true_score),
        }
    val_result.update({
        'lcc_nomapping': get_lcc(pred_score, true_score),
        'srcc_nomapping': get_srcc(pred_score, true_score),
    })
    pred_score = score_mapping(pred_score, true_score)
    if pred_score is not None:
        val_result.update({
            'L1': get_l1(pred_score, true_score),
            'L2': get_l2(pred_score, true_score),
            'lcc': get_lcc(pred_score, true_score),
            'srcc': get_srcc(pred_score, true_score),
        })

    print(f'{test_or_valid_flag}: {val_result}')
    if writer is not None and epoch is not None:
        add_dict_scalars(writer, test_or_valid_flag, val_result, epoch+1)
    return val_result


def train(args, model, train_loader, device, optimizer, writer, epoch, emd=True):
    model.train()
    
    batch_losses = []
    for i, data in enumerate(tqdm(train_loader, ncols=100, postfix=f'{epoch}/{args.epochs}epoch')):
        images = data['image'].to(device)

        outputs = model(images)
        optimizer.zero_grad()

        if emd:
            labels = data['ratings'].to(device).float()
            outputs = outputs.view(-1, 10, 1)
            loss = emd_loss(labels, outputs, 2)
        else:
            labels = data['score'].to(device).float()
            loss = torch.nn.L1Loss()(labels, outputs.squeeze())
        batch_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, args.epochs, i + 1, len(train_loader), loss))
        writer.add_scalar('train/emd_loss', loss, i + epoch * len(train_loader))

    avg_loss = sum(batch_losses) / len(batch_losses)
    print('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, avg_loss))
    writer.add_scalar('train/epoch_emd_loss', avg_loss, epoch+1)

    return avg_loss

def evaluate(args, model, device, writer, epoch, emd):
    if 'ava' in args.val_dataset:
        dataset = IAADataset(annos_file=os.path.join(args.data_dir, 'AVA/annotations/AVA_test.json'), 
                             data_dir=args.data_dir, 
                             ratings=(args.stage=='ava'),
                             transform=val_transform)
        val_loader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=args.num_workers, 
                                                 drop_last=True)
        ava_result = validate(args=args,
                              model=model,
                              device=device,
                              val_loader=val_loader,
                              writer=writer,
                              epoch=epoch,
                              test_or_valid_flag='val_ava',
                              emd=emd)
        
    if 'para' in args.val_dataset:
        dataset = IAADataset(annos_file=os.path.join(args.data_dir, 'PARA/annotations/PARA_all.json'), 
                             data_dir=args.data_dir, 
                             transform=val_transform)
        val_loader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=args.num_workers, 
                                                 drop_last=True)
        validate(args=args,
                 model=model,
                 device=device,
                 val_loader=val_loader,
                 writer=writer,
                 epoch=epoch,
                 test_or_valid_flag='val_para',
                 emd=emd)
    
    if 'tad66k' in args.val_dataset:
        dataset = IAADataset(annos_file=os.path.join(args.data_dir, 'TAD66K/annotations/TAD66K_all.json'), 
                             data_dir=args.data_dir, 
                             transform=val_transform)
        val_loader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=args.num_workers, 
                                                 drop_last=True)
        validate(args=args,
                 model=model,
                 device=device,
                 val_loader=val_loader,
                 writer=writer,
                 epoch=epoch,
                 test_or_valid_flag='val_tad66k',
                 emd=emd)

    if 'aadb' in args.val_dataset:
        dataset = IAADataset(annos_file=os.path.join(args.data_dir, 'AADB/annotations/AADB_all.json'), 
                             data_dir=args.data_dir, 
                             transform=val_transform)
        val_loader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=args.num_workers, 
                                                 drop_last=True)
        validate(args=args,
                 model=model,
                 device=device,
                 val_loader=val_loader,
                 writer=writer,
                 epoch=epoch,
                 test_or_valid_flag='val_aadb',
                 emd=emd)
        
    if 'flickr-aes' in args.val_dataset:
        dataset = IAADataset(annos_file=os.path.join(args.data_dir, 'FLICKR-AES/annotations/FLICKR-AES_all.json'), 
                             data_dir=args.data_dir, 
                             transform=val_transform)
        val_loader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=args.num_workers, 
                                                 drop_last=True)
        validate(args=args,
                 model=model,
                 device=device,
                 val_loader=val_loader,
                 writer=writer,
                 epoch=epoch,
                 test_or_valid_flag='val_flickr-aes',
                 emd=emd)
        
    if 'yfcc1m' in args.val_dataset:
        dataset = IAADataset(annos_file=os.path.join(args.data_dir, 'YFCC15M/annotations/YFCC15M_test.json'), 
                             data_dir=args.data_dir, 
                             transform=val_transform)
        val_loader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=args.num_workers, 
                                                 drop_last=True)
        yfcc_result = validate(args=args,
                                model=model,
                                device=device,
                                val_loader=val_loader,
                                writer=writer,
                                epoch=epoch,
                                test_or_valid_flag='val_yfcc1m',
                                emd=emd)
            
    if 'yfcc1m_clean' in args.val_dataset:
        dataset = IAADataset(annos_file=os.path.join(args.data_dir, 'YFCC15M/annotations/YFCC15M_clean_test.json'), 
                             data_dir=args.data_dir, 
                             transform=val_transform)
        val_loader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=args.num_workers, 
                                                 drop_last=True)
        yfcc_clean_result = validate(args=args,
                                model=model,
                                device=device,
                                val_loader=val_loader,
                                writer=writer,
                                epoch=epoch,
                                test_or_valid_flag='val_yfcc1m_clean',
                                emd=emd)
        
    if args.stage == 'ava':
        return ava_result
    elif args.stage == 'yfcc':
        return yfcc_result
    elif args.stage == 'yfcc_clean':
        return yfcc_clean_result