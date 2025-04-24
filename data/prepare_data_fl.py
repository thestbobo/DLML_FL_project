from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
from collections import defaultdict
import random


def get_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_cifar100(root="./data"):
    transform = get_transform()
    return datasets.CIFAR100(root=root, train=True, download=True, transform=transform)


def split_iid(dataset, num_clients):
    """IID sharding: randomly assign equal data to each client."""
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
    """
    Non-IID sharding: each client gets data from only `nc` classes.

    Args:
        dataset: full CIFAR-100 dataset
        num_clients: number of clients
        nc: number of classes per client
        seed: for reproducibility

    Returns:
        List of Subsets, one for each client
    """
    random.seed(seed)
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    class_partitions = list(class_indices.values())
    for lst in class_partitions:
        random.shuffle(lst)

    client_data = [[] for _ in range(num_clients)]
    class_pool = list(range(100))
    random.shuffle(class_pool)

    shards_per_class = len(dataset) // 100
    shards_per_client = (nc * shards_per_class) // num_clients

    for client_id in range(num_clients):
        assigned_classes = random.sample(class_pool, nc)
        for cls in assigned_classes:
            count = shards_per_class // nc
            client_data[client_id].extend(class_indices[cls][:count])
            class_indices[cls] = class_indices[cls][count:]

    return [Subset(dataset, idxs) for idxs in client_data]


# Usage Example:
if __name__ == "__main__":
    cifar_dataset = load_cifar100()

    # IID
    iid_clients = split_iid(cifar_dataset, num_clients=100)

    # Non-IID with Nc = 5
    noniid_clients = split_noniid(cifar_dataset, num_clients=100, nc=5)

    print(f"IID Client 0 size: {len(iid_clients[0])}")
    print(f"Non-IID Client 0 size: {len(noniid_clients[0])}")
