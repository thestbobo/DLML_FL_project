import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
import random

from torchvision.datasets import CIFAR100

def get_fixed_probe_batch(n_samples=100, seed=42, device='cpu'):
    # 1. Carica il test set completo
    test_transform = get_transform()
    test_data = CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    l = len(test_data)

    # 2. Estrai un sottoinsieme fisso (es. 100 immagini)
    random.seed(seed)
    subset_idxs = random.sample(range(len(test_data)), n_samples)
    probe_subset = Subset(test_data, subset_idxs)

    # 3. Loader per un singolo batch
    probe_loader = DataLoader(probe_subset, batch_size=n_samples, shuffle=False)
    images, _ = next(iter(probe_loader))
    return images.to(device), l  # Shape: (n_samples, C, H, W)

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



def get_client_datasets(iid, num_clients, nc, seed=42):
    """
    Load CIFAR-100 dataset and split it into client datasets.
    """
    full_dataset = load_cifar100()

    if iid:
        return split_iid(full_dataset, num_clients)
    else:
        return split_noniid(full_dataset, num_clients, nc=nc, seed=seed)


def get_test_loader(batch_size):
    test_transform = get_transform()
    test_data = CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    return DataLoader(test_data, batch_size=batch_size, shuffle=False)


def split_iid(dataset, num_clients, seed=42):
    """IID sharding: randomly assign equal data to each client."""
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
