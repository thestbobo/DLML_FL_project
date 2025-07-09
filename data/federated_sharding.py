import numpy as np
import random
from torch.utils.data import Subset
from collections import defaultdict
from .cifar import load_cifar100

def split_iid(dataset, num_clients, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    num_items = len(dataset) // num_clients
    all_indices = np.random.permutation(len(dataset))
    splits = []
    for i in range(num_clients):
        start = i * num_items
        end = (i + 1) * num_items
        indices = all_indices[start:end]
        client_subset = Subset(dataset, indices)
        splits.append(client_subset)
    return splits

def split_noniid(dataset, num_clients, nc=2, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    total_samples = len(dataset)
    total_shards = num_clients * nc
    shard_size = total_samples // total_shards
    shards = []
    for lbl, idxs in class_indices.items():
        random.shuffle(idxs)
        for i in range(0, len(idxs), shard_size):
            shard = idxs[i: i + shard_size]
            shards.append(shard)
    random.shuffle(shards)
    clients = []
    for i in range(num_clients):
        start = i * nc
        end = (i + 1) * nc
        client_idxs = [idx for shard in shards[start:end] for idx in shard]
        clients.append(Subset(dataset, client_idxs))
    return clients

def get_client_datasets(iid, num_clients, nc, seed=42):
    """
    Return a list of Subset datasets for each client.
    """
    full_dataset = load_cifar100(train=True)
    if iid:
        return split_iid(full_dataset, num_clients, seed=seed)
    else:
        return split_noniid(full_dataset, num_clients, nc=nc, seed=seed)