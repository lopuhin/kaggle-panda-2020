import torchvision.models


class ResNet(nn.Module):
    def __init__(self, n_outputs: int, name: str = 'resnet34'):
        self.base = getattr(torchvision.models, name)(pretrained=True)
        self.base.fc = nn.Linear(
            in_features=self.base.fc.in_features,
            out_features=n_outputs,
            bias=True)
    
    def forward(self, x):
        return self.base(x)
