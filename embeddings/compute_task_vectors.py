import yaml
import os

# Load paths from config/config.yaml
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


import yaml
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


import torch
import os
from torch.utils.data import Subset, DataLoader
from models.dino_ViT_b16 import DINO_ViT
from models.DeiT_ViT import DeiT_ViT
from models.MAE_ViT import MAE_ViT
from data.data_utils import load_cifar100_dataset, get_data_loader, get_split_indices, save_split_indices

from tqdm import tqdm

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.to(device)
    model.eval()
    features = []
    for inputs, _ in tqdm(loader, desc="[task_vector] Extracting embeddings"):
        inputs = inputs.to(device)
        feats = model.model(inputs)
        features.append(feats.cpu())
    return torch.cat(features, dim=0)

def compute_task_vector(model_name, finetuned_weights_path, save_path='cfg["paths"]["task_vectors_dir"]'):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('cfg["paths"]["splits_dir"]', exist_ok=True)

    print(f"[task_vector] Starting task vector computation for model: {model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[task_vector] Using device: {device}")

    # Load correct model architecture
    if model_name == 'dino':
        model = DINO_ViT(num_classes=100, pretrained=True, frozen_backbone=True)
    elif model_name == 'deit':
        model = DeiT_ViT(num_classes=100, pretrained=True, frozen_backbone=True)
    elif model_name == 'mae':
        model = MAE_ViT(num_classes=100, pretrained=True, frozen_backbone=True)
    else:
        raise ValueError("Unsupported model name")

    # Prepare data
    dataset = load_cifar100_dataset(train=True)
    if not os.path.exists('cfg["paths"]["splits_dir"]/train_idx.pt'):
        print("[task_vector] Split not found, generating...")
        train_idx, val_idx, test_idx = get_split_indices(len(dataset))
        save_split_indices(train_idx, val_idx, test_idx)
        print("[task_vector] Saved splits.")
    else:
        print("[task_vector] Using existing splits.")

    train_idx = torch.load('cfg["paths"]["splits_dir"]/train_idx.pt')
    loader = get_data_loader(dataset, train_idx, batch_size=64)

    # Extract embeddings from base model
    print("[task_vector] Extracting base embeddings...")
    base_embeddings = extract_embeddings(model, loader, device)

    # Load fine-tuned weights and extract new embeddings
    print(f"[task_vector] Loading fine-tuned weights from {finetuned_weights_path}")
    model.load_state_dict(torch.load(finetuned_weights_path, map_location=device))
    print("[task_vector] Extracting fine-tuned embeddings...")
    tuned_embeddings = extract_embeddings(model, loader, device)

    if base_embeddings.shape != tuned_embeddings.shape:
        raise ValueError(f"Shape mismatch: base={base_embeddings.shape}, tuned={tuned_embeddings.shape}")

    delta = tuned_embeddings - base_embeddings
    save_file = os.path.join(save_path, f"{model_name}_task_vector.pt")
    torch.save(delta, save_file)
    print(f"[task_vector] Saved task vector to: {save_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model name: dino, deit, or mae")
    parser.add_argument('--weights', type=str, required=True, help="Path to fine-tuned model weights")
    args = parser.parse_args()

    compute_task_vector(args.model, args.weights)


