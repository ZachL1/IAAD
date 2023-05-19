import os
from functools import cmp_to_key
import torch


def sort_ckpts(ckptl:str, ckptr:str):
    epochl = int(ckptl.split('-')[1][:-4])
    epochr = int(ckptr.split('-')[1][:-4])
    return epochl - epochr

def save_ckpt(args, model_without_ddp, optimizer, epoch, best_metrics):
    save_path = os.path.join(args.ckpt_path, 'epoch-%d.pth' % (epoch + 1))
    print('Saving model to ', save_path)
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    try:
        torch.save({
            'model': model_without_ddp.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_metrics': best_metrics,
        }, save_path)
    except OSError:
        ckpt_files = sorted(os.listdir(os.path.dirname(args.resume)), key=cmp_to_key(sort_ckpts))
        if len(ckpt_files) == 0:
            print('[WARRING] can not make more space')
            return
        to_remove = os.path.join(args.resume[:args.resume.rfind('/')], ckpt_files[0])
        print('not space on device, remove ', to_remove)
        os.system(f'rm -f {to_remove}')
        return save_ckpt(args, model_without_ddp, optimizer, epoch, best_metrics)
    else:
        print('Done.\n')