import torch
from torch import nn
import torchvision.models


N_CLASSES = 6


class ResNet(nn.Module):
    def __init__(self, n_outputs: int, name: str, pretrained: bool):
        super().__init__()
        self.base = getattr(torchvision.models, name)(pretrained=pretrained)
        self.base.fc = nn.Linear(
            in_features=2 * self.base.fc.in_features,
            out_features=n_outputs,
            bias=True)
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
        x = self.base.fc(x)
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


def resnet34(pretrained: bool = True):
    return ResNet(name='resnet34', n_outputs=N_CLASSES, pretrained=pretrained)
