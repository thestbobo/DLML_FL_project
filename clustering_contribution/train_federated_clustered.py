# 📁 train_federated_clustered.py – Clustered Conflict-Aware Editing Only

import os
import copy
import numpy as np
import torch
import yaml
import wandb
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.cluster import KMeans
from collections import defaultdict

from models.dino_ViT_b16 import DINO_ViT
from fl_core.client import local_train_with_mask
from fl_core.server import average_weights_fedavg
from data.prepare_data_fl import get_client_datasets, get_test_loader
from project_utils.metrics import get_metrics
from project_utils.federated_metrics import log_global_metrics

def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    all_outputs, all_labels = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            all_outputs.append(outputs)
            all_labels.append(y)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = get_metrics(all_outputs, all_labels)
    avg_loss = total_loss / len(dataloader.dataset)
    metrics["global_loss"] = avg_loss
    return metrics

def cluster_clients(client_datasets, num_clusters=5, num_classes=100):
    def compute_class_histogram(dataset):
        histogram = np.zeros(num_classes)
        for _, label in dataset:
            histogram[label] += 1
        return histogram / histogram.sum()
    features = [compute_class_histogram(ds) for ds in client_datasets]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(np.stack(features))
    return {i: int(clusters[i]) for i in range(len(client_datasets))}

def main():
    with open("config/config.yaml", encoding="utf-8") as f:
        default_config = yaml.safe_load(f)

    wandb.init(project="Federated-DINO-ViT", config=default_config)
    config = wandb.config

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_clusters = config.get("NUM_CLUSTERS", 5)
    masks_root = config.MASKS_DIR
    os.makedirs(masks_root, exist_ok=True)

    global_model = DINO_ViT(num_classes=100, pretrained=True, frozen_backbone=True)
    client_datasets = get_client_datasets(config.IID, config.NUM_CLIENTS, config.NC, config.seed)
    test_loader = get_test_loader(batch_size=config.BATCH_SIZE)

    print(">>> Clustering clients and loading cluster masks")
    with open(os.path.join(masks_root, "cluster_mapping.yaml"), "r") as f:
        cluster_mapping = yaml.safe_load(f)

    cluster_masks = {}
    for c_id in set(cluster_mapping.values()):
        mask_path = os.path.join(masks_root, f"mask_cluster_{c_id}.pt")
        cluster_masks[c_id] = torch.load(mask_path, map_location=device)

    for t_round in range(1, config.ROUNDS + 1):
        print(f"\n--- Round {t_round} ---")
        selected_clients = np.random.choice(config.NUM_CLIENTS, max(1, int(config.CLIENT_FRACTION * config.NUM_CLIENTS)), replace=False)

        local_weights, num_samples_list = [], []
        for cid in selected_clients:
            print(f"Training client {cid}")
            local_model = copy.deepcopy(global_model)
            loader = DataLoader(client_datasets[cid], batch_size=config.BATCH_SIZE, shuffle=True)
            cluster_id = cluster_mapping[str(cid)]
            shared_mask = cluster_masks[cluster_id]
            w, _, _ = local_train_with_mask(
                local_model,
                loader,
                local_steps=config.LOCAL_STEPS,
                lr=config.LR,
                device=device,
                global_masks=shared_mask
            )
            local_weights.append(w)
            num_samples_list.append(len(client_datasets[cid]))

        global_weights = average_weights_fedavg(local_weights, num_samples_list)
        global_model.load_state_dict(global_weights)
        metrics = evaluate(global_model, test_loader)
        log_global_metrics(metrics, t_round)

    print("\n[✓] Federated Training Complete")

if __name__ == "__main__":
    main()
