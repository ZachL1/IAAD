import os
import os.path as osp
import time
import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from datetime import timedelta
from iaa.utils.logger import setup_logger
from iaa.utils.comm import init_env, get_configured_optimizer_scheduler
from iaa.model.iaa_model import get_configured_iaa_model
from iaa.utils.running import load_ckpt
# from iaa.utils.do_test import do_scalecano_test_with_custom_data
from iaa.utils.mldb import load_data_info, reset_ckpt_path
# from iaa.utils.custom_data import load_from_annos, load_data

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--show-dir', help='the dir to save logs and visualization results')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--resume-from', help='the checkpoint file to resume weights from')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--launcher', choices=['None', 'pytorch', 'slurm', 'mpi'], default='None', help='job launcher')
    # parser.add_argument('--test_data_path', default='None', type=str, help='the path of test data')
    args = parser.parse_args()
    return args

def main(args):
    os.chdir(CODE_SPACE)
    cfg = Config.fromfile(args.config)
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        
    # show_dir is determined in this priority: CLI > segment in file > filename
    if args.show_dir is not None:
        # update configs according to CLI args if args.show_dir is not None
        cfg.show_dir = args.show_dir
    else:
        # use condig filename + timestamp as default show_dir if args.show_dir is None
        cfg.show_dir = osp.join('./show_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                args.timestamp)
    
    # ckpt path
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    # load data info
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info

    # # update check point info
    # reset_ckpt_path(cfg.model, data_info)
    
    # create show dir
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)
    
    # init the logger before other steps
    cfg.log_file = osp.join(cfg.show_dir, f'{args.timestamp}.log')
    logger = setup_logger(cfg.log_file)
    
    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')
    
    # init distributed env dirst, since logger depends on the dist info
    if args.launcher == 'None':
        cfg.distributed = False
    else:
        cfg.distributed = True
        init_env(args.launcher, cfg)
    logger.info(f'Distributed training: {cfg.distributed}')
    
    # dump config 
    cfg.dump(osp.join(cfg.show_dir, osp.basename(args.config)))
    
    if not cfg.distributed:
        main_worker(0, cfg, args.launcher)
    else:
        # distributed training
        mp.spawn(main_worker, nprocs=cfg.dist_params.num_gpus_per_node, args=(cfg, args.launcher))
        
def main_worker(local_rank: int, cfg: dict, launcher: str):
    if cfg.distributed:
        cfg.dist_params.global_rank = cfg.dist_params.node_rank * cfg.dist_params.num_gpus_per_node + local_rank
        cfg.dist_params.local_rank = local_rank

        torch.cuda.set_device(local_rank)
        default_timeout = timedelta(minutes=30)
        dist.init_process_group(
            backend=cfg.dist_params.backend,
            init_method=cfg.dist_params.dist_url,
            world_size=cfg.dist_params.world_size,
            rank=cfg.dist_params.global_rank,
            timeout=default_timeout)
    
    logger = setup_logger(cfg.log_file)
    # build model
    model = get_configured_iaa_model(cfg, )
    optimizer, scheduler = get_configured_optimizer_scheduler(cfg, )
    
    # load ckpt
    if "load_from" in cfg:
        model, _,  _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    if "resume_from" in cfg:
        model, optimizer, scheduler = load_ckpt(cfg.resume_from, model, optimizer, scheduler, strict_match=True)
    
    # config distributed training
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()
        
    
if __name__ == '__main__':
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp
    main(args)    