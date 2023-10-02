import torch
import torch.nn as nn
from iaa.utils.comm import get_func


class BaseModel(nn.Module):
    def __init__(self, cfg, **kwargs) -> None:
        super(BaseModel, self).__init__()
        model_type = cfg.model.type
        self.model = get_func('iaa.model.model_pipelines.' + model_type)(cfg)

    def forward(self, data):
        output = self.model(**data)

        return output

    def inference(self, data):
        with torch.no_grad():
            output = self.forward(data)
        return output
    