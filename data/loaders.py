import torch
from torch.utils.data import DataLoader, random_split, Subset
from .cifar import load_cifar100
from .transforms import test_transform

def get_cifar100_loaders(val_split, batch_size, num_workers, root="./data"):
    train_data = load_cifar100(train=True, root=root)
    test_data = load_cifar100(train=False, root=root)
    val_size = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size
    train_set, val_set = random_split(train_data, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def get_test_loader(batch_size, root="./data"):
    test_data = load_cifar100(train=False, root=root)
    return DataLoader(test_data, batch_size=batch_size, shuffle=False)

def get_sparse_loaders(full_train_dataset, calib_frac, calib_batch_size, batch_size, num_workers, seed=42):
    total = len(full_train_dataset)
    calib_size = int(total * calib_frac)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    calib_idx, train_idx = indices[:calib_size], indices[calib_size:]
    calib_loader = DataLoader(
        Subset(full_train_dataset, calib_idx),
        batch_size=calib_batch_size, shuffle=True, num_workers=num_workers
    )
    train_loader = DataLoader(
        Subset(full_train_dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return train_loader, calib_loader