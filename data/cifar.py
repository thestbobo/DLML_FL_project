from torchvision import datasets
from .transforms import train_transform, test_transform

def load_cifar100(train=True, root="./data"):
    transform = train_transform() if train else test_transform()
    return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)