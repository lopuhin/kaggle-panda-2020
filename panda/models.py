import io
from functools import partial
from pathlib import Path

try:
    from inplace_abn.abn import InPlaceABN
except ImportError:
    InPlaceABN = None
import requests
import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models

from . import gnws_resnet, gnws_resnext, bit_resnet
from .dataset import N_CLASSES
from .abn_models import resnet as abn_resnet


batch_norm_classes = (nn.BatchNorm2d, nn.GroupNorm)
if InPlaceABN is not None:
    batch_norm_classes += (InPlaceABN,)


class ResNet(nn.Module):
    def __init__(self, base: nn.Module, head_cls):
        super().__init__()
        self.base = base
        self.head = head_cls(
            in_features=self.get_features_dim(), out_features=N_CLASSES)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.frozen = False

    def get_features_dim(self):
        return self.base.fc.in_features
    
    def forward(self, x):
        batch_size, n_patches, *patch_shape = x.shape
        x = x.reshape((batch_size * n_patches, *patch_shape))
        x = self.get_features(x)
        n_features = x.shape[1]
        x = x.reshape((batch_size, n_patches, n_features, -1))
        x = x.transpose(1, 2).reshape((batch_size, n_features, -1))
        x = self.avgpool(x)
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

    def train(self, mode=True):
        super().train(mode)
        if mode and self.frozen:
            self._freeze(
                self.base.conv1,
                self.base.bn1,
                self.base.layer1,
                self.base.layer2,
                self.base.layer3,
                # without layer4
            )

    def _freeze(self, *modules):
        for module in modules:
            module.requires_grad_(False)
            for m in module.modules():
                if isinstance(m, batch_norm_classes):
                    m.eval()


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
    return ResNet(base=base, head_cls=head_cls)


resnet18 = partial(resnet, name='resnet18')
resnet34 = partial(resnet, name='resnet34')
resnet50 = partial(resnet, name='resnet50')


class ResNeXtGNWS(ResNet):
    def get_features(self, x):
        base = self.base
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool1(x)

        x = base.layer1(x)
        x = base.layer2(x)
        x = base.layer3(x)
        x = base.layer4(x)
        return x


def resnet_gnws(name: str, head_name: str, pretrained: bool = True):
    """
    https://github.com/joe-siyuan-qiao/pytorch-classification
    """
    if name.startswith('resnext'):
        base = getattr(gnws_resnext, name)()
        cls = ResNeXtGNWS
    else:
        cls = ResNet
        base = getattr(gnws_resnet, name)()
    if pretrained:
        weights_name = {
            'resnet50': 'R-50-GN-WS.pth.tar',
            'resnext50': 'X-50-GN-WS.pth.tar',
            'resnet101': 'R-101-GN-WS.pth.tar',
            'resnext101': 'X-101-GN-WS.pth.tar',
        }[name]
        state = torch.load(Path('data') / weights_name, map_location='cpu')
        base.load_state_dict({k.split('.', 1)[1]: v for k, v in state.items()})
    head_cls = globals()[head_name]
    return cls(base=base, head_cls=head_cls)


resnet50_gnws = partial(resnet_gnws, name='resnet50')
resnext50_gnws = partial(resnet_gnws, name='resnext50')
resnet101_gnws = partial(resnet_gnws, name='resnet101')
resnext101_gnws = partial(resnet_gnws, name='resnext101')


def resnet_swsl(name: str, head_name: str, pretrained: bool = True):
    if pretrained:
        base = torch.hub.load(
            'facebookresearch/semi-supervised-ImageNet1K-models', name)
    elif name == 'resnet50_swsl':
        base = torchvision.models.resnet50(pretrained=False)
    elif name == 'resnext50_32x4d_swsl':
        base = torchvision.models.resnext50_32x4d(pretrained=False)
    else:
        raise ValueError(f'model "{name}" not supported yet')
    head_cls = globals()[head_name]
    return ResNet(base=base, head_cls=head_cls)


resnet50_swsl = partial(resnet_swsl, name='resnet50_swsl')
resnext50_32x4_swsl = partial(resnet_swsl, name='resnext50_32x4d_swsl')


class ResNetTimm(ResNet):
    def get_features(self, x):
        base = self.base
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.act1(x)
        x = base.maxpool(x)

        x = base.layer1(x)
        x = base.layer2(x)
        x = base.layer3(x)
        x = base.layer4(x)

        return x


class TResNetTimm(ResNet):
    def get_features(self, x):
        return self.base.forward_features(x)

    def get_features_dim(self):
        return self.base.head.fc.in_features

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode and self.frozen:
            self._freeze(self.base.body[:-1])  # without layer4


def resnet_timm(name: str, head_name: str, pretrained: bool = True):
    base = timm.create_model(name, pretrained=pretrained)
    head_cls = globals()[head_name]
    return ResNetTimm(base=base, head_cls=head_cls)


resnet34_timm = partial(resnet_timm, name='resnet34')


def tresnet_timm(name: str, head_name: str, pretrained: bool = True):
    base = timm.create_model(name, pretrained=pretrained)
    head_cls = globals()[head_name]
    return TResNetTimm(base=base, head_cls=head_cls)


tresnet_m = partial(tresnet_timm, name='tresnet_m')
tresnet_l = partial(tresnet_timm, name='tresnet_l')


class BiTResNet(ResNet):
    def get_features(self, x):
        base = self.base
        return base.head[:2](base.body(base.root(x)))

    def get_features_dim(self):
        return self.base.head[0].num_channels


def get_bit_weights(bit_variant):
    cached = Path('data') / f'bit_{bit_variant}.npz'
    if cached.exists():
        data = cached.read_bytes()
    else:
        print('downloading BiT weights')
        response = requests.get(
            f'https://storage.googleapis.com/bit_models/{bit_variant}.npz')
        print('done')
        response.raise_for_status()
        data = response.content
        cached.write_bytes(data)
    return np.load(io.BytesIO(data))


def resnet_bit(name: str, head_name: str, pretrained: bool = True):
    base = bit_resnet.KNOWN_MODELS[name]()
    if pretrained:
        base.load_from(get_bit_weights(name))
    head_cls = globals()[head_name]
    return BiTResNet(base=base, head_cls=head_cls)


resnet50_bit = partial(resnet_bit, name='BiT-M-R50x1')
resnet50x3_bit = partial(resnet_bit, name='BiT-M-R50x3')
resnet101_bit = partial(resnet_bit, name='BiT-M-R101x1')
resnet101x3_bit = partial(resnet_bit, name='BiT-M-R101x3')
resnet152x2_bit = partial(resnet_bit, name='BiT-M-R152x2')
resnet152x4_bit = partial(resnet_bit, name='BiT-M-R152x4')


class ABNResNet(ResNet):
    def get_features(self, x):
        return self.base.forward(x)

    def get_features_dim(self):
        try:
            return self.base.mod5.block3.convs.bn3.bias.shape[0]
        except AttributeError:
            return self.base.mod5.block3.convs.bn2.bias.shape[0]


def resnet_abn(name: str, head_name: str, pretrained: bool = True):
    base = getattr(abn_resnet, 'net_' + name)()
    if pretrained:
        state = torch.load(f'data/abn_weights/{name}.pth.tar',
                           map_location='cpu')
        base.load_state_dict(
            {k[len('module.'):]: v for k, v in state['state_dict'].items()
             if 'classifier.' not in k})
    head_cls = globals()[head_name]
    return ABNResNet(base=base, head_cls=head_cls)


resnet34_abn = partial(resnet_abn, name='resnet34')
resnet50_abn = partial(resnet_abn, name='resnet50')
