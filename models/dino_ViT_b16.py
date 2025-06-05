import torch.nn as nn
import torch


class DINO_ViT(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, frozen_backbone=False):
        super(DINO_ViT, self).__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=pretrained)
        
        # FREEZA IL BACKBONE
        if frozen_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
                            nn.Dropout(p=0.3),
                            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
