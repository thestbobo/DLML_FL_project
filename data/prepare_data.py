import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_cifar100_loaders(val_split, batch_size, num_workers):
    """
    Downloads CIFAR-100, splits train/val/test, returns DataLoaders

    """
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        transforms.RandomCrop(32, padding=4),  # Random crop with padding (common for CIFAR)
        transforms.RandomHorizontalFlip(),  # Flip with 50% chance
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Random brightness, contrast, saturation, hue
        transforms.RandomErasing(p=0.25)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    # Load dataset
    train_data = datasets.CIFAR100(root="/content/DLML_FL_project/data", train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR100(root="/content/DLML_FL_project/data", train=False, download=True, transform=val_transform)

    # Train/val split
    val_size = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size

    train_set, val_set = random_split(train_data, [train_size, val_size])

    # Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
