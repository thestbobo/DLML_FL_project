import yaml
import os
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from models.dino_ViT_b16 import DINO_ViT
from models.DeiT_ViT import DeiT_ViT
from models.MAE_ViT import MAE_ViT
from data.data_utils import get_split_indices, save_split_indices

# === Load config ===
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# === Output dirs ===
os.makedirs(cfg["paths"]["embeddings_dir"], exist_ok=True)
os.makedirs(cfg["paths"]["splits_dir"], exist_ok=True)

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# === Dataset ===
dataset = CIFAR100(root='./data/raw', train=True, download=True, transform=transform)
print(f"[embeddings] Loaded CIFAR-100 with {len(dataset)} samples")

# === Ensure splits exist ===
split_files = ["train_idx.pt", "val_idx.pt", "test_idx.pt"]
if not all(os.path.exists(os.path.join(cfg["paths"]["splits_dir"], f)) for f in split_files):
    print("[embeddings] Splits not found, generating...")
    train_idx, val_idx, test_idx = get_split_indices(len(dataset))
    save_split_indices(train_idx, val_idx, test_idx)
    print(f"[embeddings] Splits saved to {cfg['paths']['splits_dir']}")
else:
    print("[embeddings] Using existing splits")

train_idx = torch.load(os.path.join(cfg["paths"]["splits_dir"], "train_idx.pt"))
loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[embeddings] Using device: {device}")

def extract_embeddings(model, loader, device):
    model.to(device)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            feats = model.model(inputs)
            embeddings.append(feats.cpu())

    return torch.cat(embeddings, dim=0)

def main():
    print("[embeddings] Extracting DINO...")
    dino = DINO_ViT(num_classes=100, pretrained=True, frozen_backbone=True)
    torch.save(extract_embeddings(dino, loader, device),
               os.path.join(cfg["paths"]["embeddings_dir"], "dino_embeddings.pt"))

    print("[embeddings] Extracting DeiT...")
    deit = DeiT_ViT(num_classes=100, pretrained=True, frozen_backbone=True)
    torch.save(extract_embeddings(deit, loader, device),
               os.path.join(cfg["paths"]["embeddings_dir"], "deit_embeddings.pt"))

    print("[embeddings] Extracting MAE...")
    mae = MAE_ViT(num_classes=100, pretrained=True, frozen_backbone=True)
    torch.save(extract_embeddings(mae, loader, device),
               os.path.join(cfg["paths"]["embeddings_dir"], "mae_embeddings.pt"))

    print("[embeddings] All embeddings saved.")

if __name__ == "__main__":
    main()
