import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def save_embedding_splits(dataset_size, val_ratio=0.1, test_ratio=0.1, save_path='./data/splits'):

    os.makedirs(save_path, exist_ok=True)

    indices = list(range(dataset_size))
    train_idx, temp_idx = train_test_split(indices, test_size=val_ratio + test_ratio, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    torch.save(train_idx, os.path.join(save_path, 'train_idx.pt'))
    torch.save(val_idx, os.path.join(save_path, 'val_idx.pt'))
    torch.save(test_idx, os.path.join(save_path, 'test_idx.pt'))


def load_embeddings(model_name, split='train', embedding_dir='./data/embeddings', split_dir='./data/splits'):
    emb_path = f"{embedding_dir}/{model_name}_embeddings.pt"
    idx_path = f"{split_dir}/{split}_idx.pt"

    emb = torch.load(emb_path)
    idx = torch.load(idx_path)
    return emb[idx]



def get_embedding_dataloader(embeddings, batch_size=128, shuffle=True):
    dataset = TensorDataset(embeddings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_client_embedding_shards(embeddings, client_indices_list):
    '''
    Takes the full embeddings tensor and a list of client-specific index lists,
    returns a list of embedding subsets (one per client).
    '''
    client_embeddings = []
    for indices in client_indices_list:
        subset = embeddings[indices]
        client_embeddings.append(subset)
    return client_embeddings
