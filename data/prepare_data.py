import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_cifar100_loaders(val_split, batch_size, num_workers):
    """
    Downloads CIFAR-100, splits train/val/test, returns DataLoaders

    """
    train_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
