from torch.utils.data import DataLoader, Dataset

def get_client_datasets(iid, num_clients, nc, seed):
    # Dummy: replace with real partitioning logic
    # Assume a dummy dataset
    datasets = [DummyDataset() for _ in range(num_clients)]
    return datasets

def get_test_loader(batch_size):
    # Dummy: replace with actual test set
    return DataLoader(DummyDataset(), batch_size=batch_size)

class DummyDataset(Dataset):
    def __init__(self, n=100):
        self.n = n

    def __getitem__(self, idx):
        import torch
        x = torch.randn(3, 32, 32)
        y = torch.randint(0, 100, (1,)).item()
        return x, y

    def __len__(self):
        return self.n