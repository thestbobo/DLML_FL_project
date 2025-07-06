import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.dropout = nn.Dropout(p=0.1)  # Add this line

    def forward(self, x):
        return self.dropout(self.model(x))
