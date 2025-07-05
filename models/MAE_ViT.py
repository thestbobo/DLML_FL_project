import torch.nn as nn
import timm

class MAE_ViT(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, frozen_backbone=False):
        super(MAE_ViT, self).__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224.mae',
            pretrained=pretrained,
            num_classes=0
        )
        if frozen_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.num_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
