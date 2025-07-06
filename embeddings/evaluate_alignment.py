import yaml
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import wandb

from embeddings.alignment_models.adapters import Adapter
from embeddings.alignment_models.backbone import Backbone
from data.embedding_manager import load_embeddings
from project_utils.embedding_metrics import log_cosine_similarity

# === Load config ===
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# === Config ===
latent_dim = 512
input_dim = 384
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[alignment] Using device: {device}")

wandb.init(
    project="embeddings_alignment_training",
    name="alignment_eval",
    config=cfg
)

# === Load test embeddings ===
E1 = load_embeddings('dino', split='test',
                     embedding_dir=cfg["paths"]["embeddings_dir"],
                     split_dir=cfg["paths"]["splits_dir"])

E2 = load_embeddings('deit', split='test',
                     embedding_dir=cfg["paths"]["embeddings_dir"],
                     split_dir=cfg["paths"]["splits_dir"])

# normalize embeddings
E1 = F.normalize(E1, p=2, dim=1)
E2 = F.normalize(E2, p=2, dim=1)


loader = DataLoader(TensorDataset(E1, E2), batch_size=batch_size)
print(f"[alignment] Loaded {len(E1)} test embedding pairs")

# === Load trained models ===
A1 = Adapter(input_dim, latent_dim).to(device)
T = Backbone(latent_dim).to(device)
B2 = Adapter(latent_dim, input_dim).to(device)

ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], "alignment", "vec2vec_epoch_100.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
A1.load_state_dict(checkpoint['A1'])
T.load_state_dict(checkpoint['T'])
B2.load_state_dict(checkpoint['B2'])

A1.eval()
T.eval()
B2.eval()

# === Evaluation loop ===
cos_total = 0
count = 0
with torch.no_grad():
    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1 = A1(x1)
        t1 = T(z1)
        x1_to_x2 = B2(t1)

        sim = F.cosine_similarity(x1_to_x2, x2, dim=1)
        cos_total += sim.sum().item()
        count += sim.size(0)

        print(f"[alignment] Batch cosine mean: {sim.mean().item():.4f}")

mean_sim = cos_total / count
log_cosine_similarity(mean_sim, tag='alignment/test')
print(f"[alignment] Mean Cosine Similarity (x1 → x2): {mean_sim:.4f}")
