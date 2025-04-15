import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path

def get_cifar100_loaders(val_split=0.1, batch_size=128, num_workers=2, data_dir=None):
    """
    Prepares CIFAR-100 loaders with DINO-specific transforms
    """
    # DINO ViT-S/16 specific transforms
    train_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet stats
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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

    # Train/val split
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