import torch.nn as nn



class SimpleDecoder(nn.Module):
    def __init__(self, cfg):
        super(SimpleDecoder, self).__init__()
        
        self.in_channels = cfg.model.backone.out_channels
        self.out_channels = cfg.model.decode_head.out_channels

        if self.out_channels == 1:
            self.score_regressor = nn.Sequential(
                nn.Dropout(p=0.75),
                nn.Linear(in_features=self.in_channels, out_features=self.out_channels),
                nn.Sigmoid()
            )
        else:
            self.distr_regressor = nn.Sequential(
                nn.Dropout(p=0.75),
                nn.Linear(in_features=self.in_channels, out_features=self.out_channels),
                nn.Softmax(dim=1)
        )
            
    def forward(self, feature, **kwargs):
        f = feature.view(feature.size(0), -1)
        
        if self.out_channels == 1:
            score = self.score_regressor(f)
            return dict(score=score)
        else:
            distr = self.distr_regressor(f)
            score = 0
            return dict(distr=distr, score=score)