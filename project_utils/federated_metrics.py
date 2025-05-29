import wandb
import torch
import numpy as np
from collections import Counter


def log_local_weight_norms(local_weights, round_num):
    """
    Logga la norma L2 dei pesi locali di ciascun client su wandb.
    """
    for i, weights in enumerate(local_weights):
        norm_sq = 0.0
        for p in weights.values():
            if p is not None and p.dtype.is_floating_point:
                norm_sq += (p ** 2).sum().item()
        l2_norm = norm_sq ** 0.5
        wandb.log({f"client_{i}/local_weight_l2": l2_norm, "round": round_num})


def log_global_weight_diff(old_weights, new_weights, round_num):
    """
    Logga la differenza L2 tra pesi globali nuovi e precedenti.
    """
    total_diff = 0.0
    total_params = 0
    for key in new_weights.keys():
        p_old = old_weights[key].cpu()
        p_new = new_weights[key].cpu()
        if p_old is not None and p_new is not None and p_old.dtype.is_floating_point:
            diff = (p_new - p_old)
            total_diff += (diff ** 2).sum().item()
            total_params += diff.numel()

    if total_params > 0:
        mean_l2_diff = (total_diff / total_params) ** 0.5
        wandb.log({"global_weight_l2_diff": mean_l2_diff, "round": round_num})


def log_client_class_distribution(client_datasets, selected_clients, t_round):
    """
    For each client in selected_clients, logs the histogram of class labels
    in its local shard.
    """
    for cid in selected_clients:
        shard = client_datasets[cid]

        # Extract labels either directly or via the underlying dataset + indices
        if hasattr(shard, "targets"):
            # e.g. a raw torchvision dataset
            labels = shard.targets
        else:
            # assume torch.utils.data.Subset
            base_ds = shard.dataset
            idxs = shard.indices

            if hasattr(base_ds, "targets"):
                full_labels = base_ds.targets
            elif hasattr(base_ds, "labels"):
                full_labels = base_ds.labels
            else:
                # fallback: iterate once over the subset
                labels = [label for _, label in shard]
                full_labels = None

            if full_labels is not None:
                labels = [full_labels[i] for i in idxs]

        # Compute distribution
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))

        wandb.log({
            f"client_{cid}_class_dist": dist,
            "round": t_round
        })


def log_client_data_volume(client_datasets, selected_clients, round_num):
    """
    Logga la quantità di dati per ciascun client e il totale.
    """
    total_samples = 0
    for cid in selected_clients:
        n = len(client_datasets[cid])
        total_samples += n
        wandb.log({f"client_{cid}/data_size": n, "round": round_num})

    wandb.log({"round_total_samples_used": total_samples, "round": round_num})


def log_model_gradient_norm(model, loss_fn, batch, device, round_num):
    """
    Calcola e logga la norma del gradiente medio su un batch di dati.
    """
    model.train()
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    model.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    wandb.log({"model_gradient_norm": total_norm, "round": round_num})


def log_local_metrics(metrics_dicts, round_num):
    """
    Logga loss e accuracy locali per ciascun client.
    """
    for metrics in metrics_dicts:
        cid = metrics["client_id"]
        wandb.log({
            f"client_{cid}/local_loss": metrics["loss"],
            f"client_{cid}/local_accuracy": metrics["accuracy"],
            "round": round_num
        })
