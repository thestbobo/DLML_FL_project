
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_vit_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

def load_cifar100_dataset(root='./data', train=True):
    transform = get_vit_transform()
    return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)

def get_split_indices(dataset_size, val_ratio=0.1, test_ratio=0.1, seed=42):
    indices = list(range(dataset_size))
    train_idx, temp_idx = train_test_split(indices, test_size=val_ratio + test_ratio, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)
    return train_idx, val_idx, test_idx

def save_split_indices(train_idx, val_idx, test_idx, save_dir='./data/splits'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_idx, os.path.join(save_dir, 'train_idx.pt'))
    torch.save(val_idx, os.path.join(save_dir, 'val_idx.pt'))
    torch.save(test_idx, os.path.join(save_dir, 'test_idx.pt'))

def get_data_loader(dataset, indices, batch_size=64, shuffle=True):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def get_full_loader(dataset, batch_size=64, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
