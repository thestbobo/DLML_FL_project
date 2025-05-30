import wandb
import torch
import numpy as np
from collections import Counter


def log_local_weight_norms(local_weights, round_num):
    """
    Log the L2 norm of each client's local model weights for a given round.

    Args:
        local_weights (list of dict): List of state_dicts from each client.
        round_num (int): Current federated round.
    """
    norms = []
    for weights in local_weights:
        total_sq = 0.0
        for param in weights.values():
            total_sq += param.norm().item() ** 2
        norms.append(total_sq ** 0.5)
    wandb.log({
        'round': round_num,
        'local_weight_norms': norms
    })


def log_global_weight_diff(prev_weights, new_weights, round_num):
    """
    Log the L2 norm of the difference between global weights before and after aggregation.

    Args:
        prev_weights (dict): Previous global state_dict.
        new_weights (dict): Updated global state_dict.
        round_num (int): Current federated round.
    """
    total_sq = 0.0
    for key in prev_weights.keys():
        pw = prev_weights[key].cpu()
        nw = new_weights[key].cpu()
        diff = pw - nw
        total_sq += diff.norm().item() ** 2
    diff_norm = total_sq ** 0.5
    wandb.log({
        'round': round_num,
        'global_weight_diff': diff_norm
    })


def log_aggregated_class_distribution(client_datasets, selected_clients, round_num):
    """
    Log an aggregated histogram of class distributions across selected clients.
    Call this every K rounds to avoid clutter.

    Args:
        client_datasets (list of Dataset): List of client datasets.
        selected_clients (list of int): Indices of chosen clients.
        round_num (int): Current federated round.
    """
    all_labels = []
    for cid in selected_clients:
        all_labels.extend([label for _, label in client_datasets[cid]])
    # Log raw labels as a WandB histogram
    wandb.log({
        'round': round_num,
        'aggregated_class_distribution': wandb.Histogram(all_labels)
    })


def log_round_info(round_num, num_clients, num_samples):
    """
    Log summary info for a federated round.

    Args:
        round_num (int): Round number.
        num_clients (int): Number of clients participated.
        num_samples (int): Total samples used.
    """
    wandb.log({
        'round': round_num,
        'clients_participated': num_clients,
        'samples_used': num_samples
    })


def log_client_metrics(client_id, loss, accuracy, round_num):
    """
    Log local loss and accuracy for a specific client.

    Args:
        client_id (int): Client identifier.
        loss (float): Local training loss.
        accuracy (float): Local training accuracy.
        round_num (int): Current federated round.
    """
    wandb.log({
        'round': round_num,
        f'client_{client_id}/local_loss': loss,
        f'client_{client_id}/local_accuracy': accuracy
    })


def log_global_metrics(metrics, round_num):
    """
    Log global evaluation metrics after aggregation.

    Args:
        metrics (dict): Keys are metric names, values are metric values.
        round_num (int): Current federated round.
    """
    data = {'round': round_num}
    for key, value in metrics.items():
        data[f'global_test_{key}'] = value
    wandb.log(data)
