import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from project_utils.embedding_metrics import log_cosine_similarity
from data.data_utils import load_cifar100_dataset, get_split_indices, save_split_indices
import os

def evaluate_transfer(base_path, edited_path, target_path, batch_size=128):
    """
    Compare base vs edited embeddings in terms of alignment to target embeddings.
    """
    print("[transfer] Loading embeddings...")
    base = torch.load(base_path)
    edited = torch.load(edited_path)
    target = torch.load(target_path)

    assert base.shape == edited.shape == target.shape, "[transfer] Shape mismatch between embeddings"

    os.makedirs('./data/splits', exist_ok=True)
    if not os.path.exists('./data/splits/test_idx.pt'):
        print("[transfer] Generating missing split indices...")
        dataset = load_cifar100_dataset(train=True)
        train_idx, val_idx, test_idx = get_split_indices(len(dataset))
        save_split_indices(train_idx, val_idx, test_idx)
        print("[transfer] Saved split indices")
    else:
        print("[transfer] Using existing test split")

    test_idx = torch.load('./data/splits/test_idx.pt')
    base_loader = DataLoader(TensorDataset(base[test_idx], target[test_idx]), batch_size=batch_size)
    edited_loader = DataLoader(TensorDataset(edited[test_idx], target[test_idx]), batch_size=batch_size)

    def avg_cosine(loader, tag):
        total_sim = 0
        count = 0
        for x, y in loader:
            sim = F.cosine_similarity(x, y, dim=1)
            total_sim += sim.sum().item()
            count += sim.size(0)
        mean_sim = total_sim / count
        log_cosine_similarity(mean_sim, tag=tag)
        return mean_sim

    print("[transfer] Evaluating cosine similarities...")
    base_score = avg_cosine(base_loader, tag='transfer/base_vs_target')
    edited_score = avg_cosine(edited_loader, tag='transfer/edited_vs_target')

    print(f"[transfer] Base Cosine Similarity:   {base_score:.4f}")
    print(f"[transfer] Edited Cosine Similarity: {edited_score:.4f}")
    print(f"[transfer] Delta Improvement:        {edited_score - base_score:.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=True, help="Path to original embeddings")
    parser.add_argument('--edited', type=str, required=True, help="Path to embeddings after vector applied")
    parser.add_argument('--target', type=str, required=True, help="Path to real fine-tuned embeddings")
    args = parser.parse_args()

    evaluate_transfer(args.base, args.edited, args.target)

