import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path

def get_cifar100_loaders(val_split=0.1, batch_size=128, num_workers=2, data_dir=None):
    """
    Prepares CIFAR-100 loaders with ViT-optimized transforms
    Args:
        val_split: Fraction of training data to use for validation
        batch_size: Batch size for all loaders
        num_workers: Number of workers for data loading
        data_dir: Directory to store/lookup dataset
    Returns:
        train_loader, val_loader, test_loader
    """
    # ViT-S/16 optimized transforms
    train_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        transforms.RandomErasing(p=0.25)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    # Set data directory
    data_path = Path(data_dir) if data_dir else Path("./data")
    data_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_data = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform
    )
    test_data = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=val_transform
    )

    # Train/val split with fixed random seed for reproducibility
    val_size = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size
    train_set, val_set = random_split(
        train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader, test_loader