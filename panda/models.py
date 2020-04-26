from torch import nn
import torchvision.models


N_CLASSES = 6


class ResNet(nn.Module):
    def __init__(self, n_outputs: int, name: str):
        super().__init__()
        self.base = getattr(torchvision.models, name)(pretrained=True)
        self.base.fc = nn.Linear(
            in_features=self.base.fc.in_features,
            out_features=n_outputs,
            bias=True)
    
    def forward(self, x):
        return self.base(x)


def resnet34():
    return ResNet(name='resnet34', n_outputs=N_CLASSES)
