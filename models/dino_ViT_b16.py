import torch.nn as nn
import torch
from torchvision.models import vit_b_16

class DINO_ViT(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, unfreeze_last_block=False):
        super(DINO_ViT, self).__init__()
        
        # Load pretrained DINO ViT-S/16
        self.model = vit_b_16(weights='DEFAULT' if pretrained else None)
        
        # Freeze all layers by default
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze last transformer block if specified
        if unfreeze_last_block:
            for param in self.model.encoder.layers[-1].parameters():
                param.requires_grad = True
        
        # Replace classification head
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)