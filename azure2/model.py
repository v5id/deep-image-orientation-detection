import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class OrientationNet(nn.Module):

    def __init__(self, weights=MobileNet_V3_Small_Weights.DEFAULT):
        super().__init__()

        self.model = mobilenet_v3_small(
            weights=weights
        )

        # freeze backbone
        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier[3] = nn.Linear(1024, 4)

    def forward(self, x):
        return self.model(x)
