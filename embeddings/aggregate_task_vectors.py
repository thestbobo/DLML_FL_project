
import yaml
import os
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


import torch

def aggregate_task_vectors(model_names, method='mean'):
    aligned_dir = cfg["paths"]["aligned_vectors_dir"]
    save_path = cfg["paths"]["aggregated_vector_path"]

    vectors = []
    for name in model_names:
        path = os.path.join(aligned_dir, f"{name}_task_vector_aligned.pt")
        vec = torch.load(path)
        vectors.append(vec)

    stacked = torch.stack(vectors)
    if method == 'mean':
        aggregated = stacked.mean(dim=0)
    elif method == 'sum':
        aggregated = stacked.sum(dim=0)
    else:
        raise ValueError("Unsupported aggregation method")

    torch.save(aggregated, save_path)
    print(f"[✓] Aggregated vector saved to {save_path} using method: {method}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help="List of model names (e.g., dino deit mae)")
    parser.add_argument('--method', type=str, default='mean', help="Aggregation method: mean or sum")
    args = parser.parse_args()

    aggregate_task_vectors(args.models, method=args.method)

