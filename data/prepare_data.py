import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np

def get_cifar100_loaders(config):
   
    # Use config values with defaults
    data_dir = config.get("data_dir", "./data")
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 2)
    val_split = config.get("val_split", 0.1)
    image_size = config.get("image_size", 224)
    resize_size = config.get("resize_size", 256)


    train_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets with file caching if available
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Use uint8 storage for memory efficiency
    train_data = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform,
        target_transform=lambda y: torch.tensor(y, dtype=torch.long)
    )

    test_data = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=val_transform,
        target_transform=lambda y: torch.tensor(y, dtype=torch.long)
    )

    # train/val split
    val_size = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size
    train_set, val_set = random_split(
        train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  
    )

    # DataLoader configuration
    loader_args = {
        'batch_size': batch_size,
        'pin_memory': True,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        num_workers=min(num_workers, 4),  # Cap workers for Colab
        **loader_args
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        num_workers=min(num_workers, 2),  # Fewer workers for validation
        **loader_args
    )

    test_loader = DataLoader(
        test_data,
        shuffle=False,
        num_workers=min(num_workers, 2),
        **loader_args
    )

    # Warmup the dataloaders (avoids initial latency)
    if num_workers > 0:
        _ = next(iter(train_loader))
        _ = next(iter(val_loader))

    return train_loader, val_loader, test_loader