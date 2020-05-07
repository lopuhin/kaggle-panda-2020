from pathlib import Path
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models

from . import gnws_resnet


class ResNet(nn.Module):
    def __init__(self, base: nn.Module, head_cls):
        super().__init__()
        self.base = base
        self.head = head_cls(
            in_features=2 * self.base.fc.in_features, out_features=1)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
    
    def forward(self, x):
        batch_size, n_patches, *patch_shape = x.shape
        x = x.reshape((batch_size * n_patches, *patch_shape))
        x = self.get_features(x)
        n_features = x.shape[1]
        x = x.reshape((batch_size, n_patches, n_features, -1))
        x = x.transpose(1, 2).reshape((batch_size, n_features, -1))
        x = torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x.squeeze(1)

    def get_features(self, x):
        base = self.base
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)

        x = base.layer1(x)
        x = base.layer2(x)
        x = base.layer3(x)
        x = base.layer4(x)
        return x


class HeadFC(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True)

    def forward(self, x):
        return self.fc(x)


class HeadFC2(nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            hidden_size: int = 512):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=hidden_size,
            bias=True)
        self.gn = nn.GroupNorm(32, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(
            in_features=hidden_size,
            out_features=out_features,
            bias=True)

    def forward(self, x):
        # x = linear_ws(self.fc, x)
        x = self.fc(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x


def linear_ws(layer, x):
    """ Adapted from https://github.com/joe-siyuan-qiao/WeightStandardization/
    """
    weight = layer.weight
    weight_mean = weight.mean(dim=1, keepdim=True)
    weight = weight - weight_mean
    std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1) + 1e-5
    weight = weight / std.expand_as(weight)
    return F.linear(x, weight, layer.bias)


def resnet(name: str, head_name: str, pretrained: bool = True):
    base = getattr(torchvision.models, name)(pretrained=pretrained)
    head_cls = globals()[head_name]
    return ResNet(
        base=base,
        head_cls=head_cls,
    )


resnet18 = partial(resnet, name='resnet18')
resnet34 = partial(resnet, name='resnet34')
resnet50 = partial(resnet, name='resnet50')


def resnet_gnws(name: str, head_name: str, pretrained: bool = True):
    """
    https://github.com/joe-siyuan-qiao/pytorch-classification
    """
    base = getattr(gnws_resnet, name)()
    if pretrained:
        weights_name = {
            'resnet50': 'R-50-GN-WS.pth.tar',
        }[name]
        state = torch.load(Path('data') / weights_name, map_location='cpu')
        base.load_state_dict({k.split('.', 1)[1]: v for k, v in state.items()})
    head_cls = globals()[head_name]
    return ResNet(
        base=base,
        head_cls=head_cls,
    )


resnet50_gnws = partial(resnet_gnws, name='resnet50')


def resnet_swsl(name: str, head_name: str, pretrained: bool = True):
    # TODO kaggle support
    base = torch.hub.load(
        'facebookresearch/semi-supervised-ImageNet1K-models', name)
    head_cls = globals()[head_name]
    return ResNet(
        base=base,
        head_cls=head_cls,
    )


resnet50_swsl = partial(resnet_swsl, name='resnet50_swsl')
resnext50_32x4_swsl = partial(resnet_swsl, name='resnext50_32x4d_swsl')
