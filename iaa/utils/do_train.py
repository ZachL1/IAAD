import torch
import logging
import os
import os.path as osp








def do_train_with_cfg(
    model: torch.nn.Module,
    cfg: dict,
    logger: logging.RootLogger,
    is_distributed: bool = True,
    local_rank: int = 0,
):
    
    
    train_data = 