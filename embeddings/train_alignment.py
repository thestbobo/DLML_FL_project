import wandb
import yaml
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import os
import random
import numpy as np

from embeddings.alignment_models.adapters import Adapter
from embeddings.alignment_models.backbone import Backbone
from embeddings.alignment_models.discriminator import Discriminator
from project_utils.embedding_metrics import (
    adversarial_loss,
    reconstruction_loss,
    cycle_consistency_loss,
    vector_space_preservation
)
from data.embedding_manager import load_embeddings
from project_utils.embedding_metrics import log_alignment_losses


with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# reproducibility
seed = cfg["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration
latent_dim = 512
input_dim = 384
batch_size = 128
num_epochs = 200
lr = 1e-4
lambda_rec = 1.0
lambda_cc = 1.0
lambda_vsp = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(
    project="embeddings_alignment_training",
    name=f"alignment_latent_{latent_dim}",
    config={
        "latent_dim": latent_dim,
        "input_dim": input_dim,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "lambda_rec": lambda_rec,
        "lambda_cc": lambda_cc,
        "lambda_vsp": lambda_vsp
    }
)

# Load embeddings
E1 = load_embeddings('dino', split='train',
                     embedding_dir=cfg["paths"]["embeddings_dir"],
                     split_dir=cfg["paths"]["splits_dir"])

E2 = load_embeddings('deit', split='train',
                     embedding_dir=cfg["paths"]["embeddings_dir"],
                     split_dir=cfg["paths"]["splits_dir"])

# normalize embeddings
E1 = F.normalize(E1, p=2, dim=1)
E2 = F.normalize(E2, p=2, dim=1)

dataset = TensorDataset(E1, E2)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define models
A1 = Adapter(input_dim, latent_dim).to(device)
A2 = Adapter(input_dim, latent_dim).to(device)
T = Backbone(latent_dim).to(device)
B1 = Adapter(latent_dim, input_dim).to(device)
B2 = Adapter(latent_dim, input_dim).to(device)

D1 = Discriminator(input_dim).to(device)
D2 = Discriminator(input_dim).to(device)
DL1 = Discriminator(latent_dim).to(device)
DL2 = Discriminator(latent_dim).to(device)

# Optimizers
gen_params = list(A1.parameters()) + list(A2.parameters()) + list(T.parameters()) + list(B1.parameters()) + list(B2.parameters())
opt_G = torch.optim.Adam(gen_params, lr=lr)
opt_D = torch.optim.Adam(list(D1.parameters()) + list(D2.parameters()) + list(DL1.parameters()) + list(DL2.parameters()), lr=lr)

# schedulers
scheduler_G = CosineAnnealingLR(opt_G, T_max=num_epochs)
scheduler_D = CosineAnnealingLR(opt_D, T_max=num_epochs)

# cosine loss function
cosine_loss_fn = torch.nn.CosineEmbeddingLoss()
lambda_cos = 1.0

# Training loop
for epoch in range(num_epochs):
    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)

        # === === Forward for Discriminator === ===
        with torch.no_grad():
            z1_d = A1(x1)
            z2_d = A2(x2)
            t1_d = T(z1_d)
            t2_d = T(z2_d)
            x1_to_x2_d = B2(t1_d)
            x2_to_x1_d = B1(t2_d)

        # === Train Discriminators ===
        opt_D.zero_grad()
        d1_loss = adversarial_loss(D1(x1), True) + adversarial_loss(D1(x2_to_x1_d.detach()), False)
        d2_loss = adversarial_loss(D2(x2), True) + adversarial_loss(D2(x1_to_x2_d.detach()), False)
        dl1_loss = adversarial_loss(DL1(z1_d), True) + adversarial_loss(DL1(t2_d.detach()), False)
        dl2_loss = adversarial_loss(DL2(z2_d), True) + adversarial_loss(DL2(t1_d.detach()), False)
        d_loss = d1_loss + d2_loss + dl1_loss + dl2_loss
        d_loss.backward()
        opt_D.step()
        scheduler_D.step()

        # === === Forward for Generator === ===
        z1 = A1(x1)
        z2 = A2(x2)
        t1 = T(z1)
        t2 = T(z2)
        x1_to_x2 = B2(t1)
        x2_to_x1 = B1(t2)

        # === Train Generators ===
        opt_G.zero_grad()
        g_adv = (
            adversarial_loss(D1(x2_to_x1), True) +
            adversarial_loss(D2(x1_to_x2), True) +
            adversarial_loss(DL1(t2), True) +
            adversarial_loss(DL2(t1), True)
        )
        rec = reconstruction_loss(x1, B1(t1)) + reconstruction_loss(x2, B2(t2))
        cyc = cycle_consistency_loss(x1, B1(T(A2(x1_to_x2.detach())))) + \
              cycle_consistency_loss(x2, B2(T(A1(x2_to_x1.detach()))))
        vsp = vector_space_preservation(x1, x1_to_x2.detach()) + \
              vector_space_preservation(x2, x2_to_x1.detach())

        target = torch.ones(x1.size(0), device=device)  # Label = 1 for similar pairs
        cos_loss = cosine_loss_fn(x1_to_x2, x2) + cosine_loss_fn(x2_to_x1, x1)

        g_loss = g_adv + lambda_rec * rec + lambda_cc * cyc + lambda_vsp * vsp + lambda_cos * cos_loss

        g_loss.backward()
        opt_G.step()
        scheduler_G.step()

    # Logging
    log_alignment_losses({
        'g_loss': g_loss.item(),
        'd_loss': d_loss.item(),
        'rec_loss': rec.item(),
        'cyc_loss': cyc.item(),
        'vsp_loss': vsp.item(),
        'cosine_loss': cos_loss.item()
    }, step=epoch)

    print(f"[Epoch {epoch+1}] Generator Loss: {g_loss.item():.4f} | Discriminator Loss: {d_loss.item():.4f}")

    if (epoch + 1) % 10 == 0:
        save_dir = os.path.join(cfg["paths"]["checkpoints_dir"], "alignment")
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'A1': A1.state_dict(), 'A2': A2.state_dict(),
            'T': T.state_dict(),
            'B1': B1.state_dict(), 'B2': B2.state_dict()
        }, os.path.join(save_dir, f'vec2vec_epoch_{epoch+1}.pt'))


wandb.finish()