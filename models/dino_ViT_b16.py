import torch.nn as nn
import torch

from torchvision.models import ViT_B_16_Weights     # might have to manually download from the GitHub repo (facebook research) the original DINOViT weights

class DINO_ViT(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(DINO_ViT, self).__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        # self.model = vit_b_16(weights='DEFAULT' if pretrained else None)
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

