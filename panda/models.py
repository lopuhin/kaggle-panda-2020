import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models


N_CLASSES = 6


class ResNet(nn.Module):
    def __init__(self, n_outputs: int, name: str, head_cls, pretrained: bool):
        super().__init__()
        self.base = getattr(torchvision.models, name)(pretrained=pretrained)
        self.head = head_cls(
            in_features=2 * self.base.fc.in_features,
            out_features=n_outputs,
        )
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
        return x

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
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(
            in_features=hidden_size,
            out_features=out_features,
            bias=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x


def resnet34(head_name: str, pretrained: bool = True):
    head_cls = globals()[head_name]
    return ResNet(
        name='resnet34',
        head_cls=head_cls,
        n_outputs=N_CLASSES,
        pretrained=pretrained,
    )
