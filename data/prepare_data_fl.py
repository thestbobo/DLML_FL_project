import random
import numpy as np

from collections import defaultdict
from torchvision.datasets import CIFAR100
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader


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

""" 
helper function to reduce the amount of data shown to the clients 
in testing, dataset size is reduced by the same proportion that number of clients is reduced
therefore, the clients see the same amount of data
"""

def downsample_stratified(dataset, frac, seed=42):
    random.seed(seed)
    # group indices by label
    class_idxs = defaultdict(list)
    for idx, (_, lbl) in enumerate(dataset):
        class_idxs[lbl].append(idx)
    # sample frac of each class
    keep = []
    for lbl, idxs in class_idxs.items():
        k = max(1, int(len(idxs) * frac))
        keep += random.sample(idxs, k)
    return Subset(dataset, keep)

def get_client_datasets(iid, num_clients, nc, reduction_frac=None, seed=42):
    """
    Load CIFAR-100 dataset and split it into client datasets.
    """
    full_dataset = load_cifar100()

    if reduction_frac is not None:
        full_dataset = downsample_stratified(full_dataset, reduction_frac, seed)

    if iid:
        return split_iid(full_dataset, num_clients)
    else:
        return split_noniid(full_dataset, num_clients, nc=nc, seed=seed)

def get_test_loader(batch_size):
    test_transform = get_transform()
    test_data = CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    return DataLoader(test_data, batch_size=batch_size, shuffle=False)

def split_iid(dataset, num_clients, seed=42):
    """
    IID sharding: randomly assign equal data to each client.
    """
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
