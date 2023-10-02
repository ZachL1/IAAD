import torch
import torch.nn as nn
from .model_pipelines.__base_model__ import BaseModel

class IAAModel(BaseModel):
    def __init__(self, cfg, **kwards):
        super(IAAModel, self).__init__(cfg)   
        model_type = cfg.model.type
        
    def inference(self, data):
        with torch.no_grad():
            output_dict = self.forward(data)       
        return output_dict


def get_configured_iaa_model(
    cfg: dict,
    ) -> nn.Module:
    """
        Args:
        @ configs: configures for the network.
        @ load_imagenet_model: whether to initialize from ImageNet-pretrained model.
        @ imagenet_ckpt_fpath: string representing path to file with weights to initialize model with.
        Returns:
        # model: iaa model.
    """
    # config iaa  model
    model = IAAModel(cfg)
    #model.init_weights(load_imagenet_model, imagenet_ckpt_fpath)
    assert isinstance(model, nn.Module)
    return model
