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

    def forward(self, x):
        return self.model(x)
