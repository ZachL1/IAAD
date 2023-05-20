from __future__ import print_function, division
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

# def adjust_learning_rate(params, optimizer, epoch):
#     """Sets the learning rate to the initial LR
#        decayed by 10 every 30 epochs"""
#     lr = params.init_lr * (0.1 ** (epoch // 10))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        #avgpool
        self.avgpool = nn.AvgPool2d(input_size // 32)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def resnet365_backbone():
    arch = 'resnet18'
    # load the pre-trained weights
    model_file = 'pretrained/resnet18_places365.pth.tar'
    last_model = models.__dict__[arch](num_classes=365)

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    last_model.load_state_dict(state_dict)
    return last_model

def mobile_net_v2(pretrained=False):
    model = MobileNetV2()

    if pretrained:
        print("read mobilenet weights")
        path_to_model = 'pretrained/mobilenetv2.pth.tar'
        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
    return model

def Attention(x):
    batch_size, in_channels, h, w = x.size()
    quary = x.view(batch_size, in_channels, -1)
    key = quary
    quary = quary.permute(0, 2, 1)

    sim_map = torch.matmul(quary, key)

    ql2 = torch.norm(quary, dim=2, keepdim=True)
    kl2 = torch.norm(key, dim=1, keepdim=True)
    sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))

    return sim_map

def MV2():
    model = mobile_net_v2()
    model = nn.Sequential(*list(model.children())[:-1])
    return model

class L5(nn.Module):
    def __init__(self):
        super(L5, self).__init__()
        back_model = MV2()
        self.base_model = back_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class L1(nn.Module):

    def __init__(self):
        super(L1, self).__init__()

        self.last_out_w = nn.Linear(365, 100)
        self.last_out_b = nn.Linear(365, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, x):
        res_last_out_w = self.last_out_w(x)
        res_last_out_b = self.last_out_b(x)
        param_out = {}
        param_out['res_last_out_w'] = res_last_out_w
        param_out['res_last_out_b'] = res_last_out_b
        return param_out

# L3
class TargetNet(nn.Module):

    def __init__(self, emd=True):
        super(TargetNet, self).__init__()
        self.emd = emd
        # L2
        self.fc1 = nn.Linear(365, 100)
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)
        self.bn1 = nn.BatchNorm1d(100).cuda()
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(1 - 0.5)

        self.relu7 = nn.PReLU()
        self.relu7.cuda()
        self.sig = nn.Sigmoid()
        if not self.emd:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x, paras):

        q = self.fc1(x)
        # print(q.shape)
        q = self.bn1(q)
        q = self.relu1(q)
        q = self.drop1(q)

        self.lin = nn.Sequential(TargetFC(paras['res_last_out_w'], paras['res_last_out_b']))
        q = self.lin(q)
        if self.emd:
            bn7 = nn.BatchNorm1d(q.shape[0])
            bn7.cuda()
            q = bn7(q)
            q = self.relu7(q)
        else:
            q = self.softmax(q)
        return q

class TargetFC(nn.Module):
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):
        out = F.linear(input_, self.weight, self.bias)
        return out

class TANet(nn.Module):
    def __init__(self, emd=True):
        super(TANet, self).__init__()
        self.emd = emd

        self.res365_last = resnet365_backbone()
        self.hypernet = L1()

        # L3
        self.tygertnet = TargetNet(self.emd)

        self.avg = nn.AdaptiveAvgPool2d((10, 1))
        self.avg_RGB = nn.AdaptiveAvgPool2d((12, 12))

        self.mobileNet = L5()
        self.softmax = nn.Softmax(dim=1)

        # L4
        self.head_rgb = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(20736, 10),
            nn.Softmax(dim=1)
        )

        # L6
        if self.emd:
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=0.75),
                nn.Linear(30, 10),
                nn.Softmax(dim=1)
            )
        else:
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=0.75),
                nn.Linear(30, 1),
                nn.Sigmoid()
            )

    def forward(self, x):

        x_temp = self.avg_RGB(x)
        x_temp = Attention(x_temp)
        x_temp =x_temp.view(x_temp.size(0),-1)
        x_temp = self.head_rgb(x_temp)

        res365_last_out = self.res365_last(x)
        res365_last_out_weights = self.hypernet(res365_last_out)
        res365_last_out_weights_mul_out = self.tygertnet(res365_last_out, res365_last_out_weights)

        x2 = res365_last_out_weights_mul_out.unsqueeze(dim=2)
        x2 = self.avg(x2)
        x2 = x2.squeeze(dim=2)


        x1 = self.mobileNet(x)

        x = torch.cat([x1,x2,x_temp],1)
        x = self.head(x)

        return x
