import torch
import torch.nn as nn
import torch.hub

class DINO_ViT(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, unfreeze_last_block=False):
        super().__init__()
        
        # Load original DINO ViT-S/16
        self.vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=pretrained)
        
        # Freeze all parameters by default
        for param in self.vits16.parameters():
            param.requires_grad = False
            
        # Unfreeze last block if specified
        if unfreeze_last_block:
            for param in self.vits16.blocks[-1].parameters():
                param.requires_grad = True
        
        # Replace classification head
        in_features = self.vits16.embed_dim
        self.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # Forward pass through DINO ViT
        features = self.vits16(x)
        return self.head(features)