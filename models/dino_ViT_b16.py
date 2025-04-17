import torch.nn as nn
import torch

from torchvision.models import ViT_B_16_Weights     # might have to manually download from the GitHub repo (facebook research) the original DINOViT weights

class DINO_ViT(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(DINO_ViT, self).__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=pretrained)
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
