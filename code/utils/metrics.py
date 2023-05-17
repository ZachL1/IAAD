import torch
import numpy as np

from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score

def get_score(y_pred:torch.Tensor, device):
    w = torch.linspace(1, 10, 10, device=device)
    w_batch = w.repeat(y_pred.size(0), 1).unsqueeze(dim=2)

    score = (y_pred * w_batch).sum(dim=1).squeeze()
    # score_np = score.data.cpu().numpy()
    return score

def get_lcc(p, gt) -> float:
    return pearsonr(p, gt)[0]

def get_srcc(p, gt) -> float:
    return spearmanr(p, gt)[0]

def get_acc(p, gt, threshold:float=5) -> float:
    p_lable = np.where(np.array(p) <= threshold, 0, 1)
    gt_lable = np.where(np.array(gt) <= threshold, 0, 1)
    return float(accuracy_score(gt_lable, p_lable))

def get_l1(p, gt) -> float:
    return np.mean(np.abs(p - gt))

def get_l2(p, gt) -> float:
    return np.mean(np.power((p - gt),2))

def score_mapping(p, gt):
    p, gt = np.array(p), np.array(gt)
    def func(x, a, b, c, d, e):
        logist = 0.5 - 1/(1+np.exp(b * (x-c)))
        return a*logist + d*x + e
    try:
        popt, pcov = curve_fit(func, p, gt)
    except RuntimeError:
        return None
    return func(p, *popt)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BestMeteric(object):
    def __init__(self):
        self.min_meteric = ['emd_loss']
        self.reset()
    def reset(self):
        self.best_meter = {}
        self.best_cnt = 0 # for early stopping

    def update(self, val_meter:dict) -> bool:
        """update val meterics

        Args:
            val_meter (dict): results of this(epoch) validation

        Returns:
            bool: Is the best validation result updated
        """
        updated = False
        for k,v in val_meter.items():
            # small is better
            if k in self.min_meteric:
                if k not in self.best_meter:
                    self.best_meter[k] = float('inf')
                if v < self.best_meter[k]:
                    self.best_meter[k] = v
                    updated = True
                    self.best_cnt = 0
            # big is better
            else:
                if k not in self.best_meter:
                    self.best_meter[k] = 0.
                if v > self.best_meter[k]:
                    self.best_meter[k] = v
                    updated = True
                    self.best_cnt = 0
        if not updated:
            self.best_cnt += 1

        return updated
    
    def best_times(self):
        return self.best_cnt
