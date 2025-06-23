import torch
import os
import numpy as np
import yaml
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from collections import defaultdict
from model_editing.TaLoS import compute_fisher_scores, calibrate_mask_layerwise_qk
from models.dino_ViT_b16 import DINO_ViT

def compute_class_histogram(dataset, num_classes=100):
    """Compute class histogram for a dataset."""
    histogram = np.zeros(num_classes)
    for _, label in dataset:
        histogram[label] += 1
    return histogram / histogram.sum()

def cluster_clients(client_datasets, num_clusters):
    features = [compute_class_histogram(ds) for ds in client_datasets]
    features = np.array(features)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    mapping = {i: int(clusters[i]) for i in range(len(client_datasets))}
    return mapping

def compute_masks_per_cluster(client_datasets, cluster_mapping, config, device):
    cluster_datasets = defaultdict(list)
    for cid, cluster_id in cluster_mapping.items():
        cluster_datasets[cluster_id].append(client_datasets[cid])

    masks_per_cluster = {}
    for cluster_id, datasets in cluster_datasets.items():
        model = DINO_ViT(num_classes=100, pretrained=True, frozen_backbone=False).to(device)
        dataset = torch.utils.data.ConcatDataset(datasets)
        loader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()

        fisher = compute_fisher_scores(model, loader, criterion, device)
        mask = calibrate_mask_layerwise_qk(
            model,
            fisher,
            keep_ratio_per_block=1.0 - config["TALOS_TARGET_SPARSITY"],
            max_rounds=config["TALOS_PRUNE_ROUNDS"]
        )
        masks_per_cluster[cluster_id] = mask
    return masks_per_cluster
