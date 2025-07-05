
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim=384, latent_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        return self.model(x)


