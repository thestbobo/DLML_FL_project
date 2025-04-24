from collections import OrderedDict
import torch

def average_weights_fedavg(weights_list, num_samples_list):
    """
    FedAvg: weighted average of model weights based on number of samples per client.

    Args:
        weights_list (list): List of client model state_dicts.
        num_samples_list (list): List of number of training samples per client.

    Returns:
        OrderedDict: Aggregated model weights.
    """
    if not weights_list:
        raise ValueError("weights_list is empty")

    total_samples = sum(num_samples_list)
    avg_weights = OrderedDict()

    for key in weights_list[0].keys():
        # Initialize tensor to zeros
        avg_weights[key] = torch.zeros_like(weights_list[0][key])

        # Weighted sum
        for weights, num_samples in zip(weights_list, num_samples_list):
            avg_weights[key] += weights[key] * (num_samples / total_samples)

    return avg_weights
