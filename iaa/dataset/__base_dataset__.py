from torch.utils import data


class BaseDataset(data.Dataset):
    def __init__(self, cfg, )