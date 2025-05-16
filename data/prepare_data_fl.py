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
    Non-IID sharding: carve the training set into num_clients*nc shards
    and give each client exactly nc shards (so each client sees only
    nc distinct classes, but the classes are mixed across shards).

    Args:
        dataset      : a torchvision Dataset with .__getitem__ returning (_, label)
        num_clients  : how many clients
        nc           : how many shards (hence classes) per client
        seed         : random seed for reproducibility

    Returns:
        List[Subset(dataset)] of length num_clients
    """
    random.seed(seed)

    # Group all indices by their class label
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # define how many shards total and compute each shard's size
    total_samples = len(dataset)
    total_shards = num_clients * nc
    shard_size = total_samples // total_shards
    shards = []

    # for each class, shuffle its indices and chop into shards of 'shard_size'
    for lbl, idxs in class_indices.items():
        random.shuffle(idxs)
        for i in range(0, len(idxs), shard_size):
            shard = idxs[i: i + shard_size]
            shards.append(shard)

    # shuffle all shards globally, then assign contiguous nc shards to each client
    random.shuffle(shards)
    clients = []
    for i in range(num_clients):
        start = i * nc
        end = (i + 1) * nc
        # flatten the list of lists into a single list of indices
        client_idxs = [idx for shard in shards[start:end] for idx in shard]
        clients.append(Subset(dataset, client_idxs))

    return clients


# Usage Example:
if __name__ == "__main__":
    cifar_dataset = load_cifar100()

    # IID
    iid_clients = split_iid(cifar_dataset, num_clients=100)

    # Non-IID with Nc = 5
    non_iid_clients = split_noniid(cifar_dataset, num_clients=100, nc=5)

    print(f"IID Client 0 size: {len(iid_clients[0])}")
    print(f"Non-IID Client 0 size: {len(non_iid_clients[0])}")
