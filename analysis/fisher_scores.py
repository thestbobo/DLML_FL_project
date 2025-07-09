import torch
import matplotlib.pyplot as plt
import numpy as np

# 1) Load your Fisher‐score dict
fisher = torch.load('analysis/fisher_global.pt', map_location='cpu')

# 2) Flatten all scores into one long 1D array
all_scores = torch.cat([v.flatten() for v in fisher.values()]).cpu().numpy()

# 3) Compute summary statistics
stats = {
    'count'   : all_scores.size,
    'min'     : float(all_scores.min()),
    '25th %'  : float(np.percentile(all_scores, 25)),
    'median'  : float(np.median(all_scores)),
    '75th %'  : float(np.percentile(all_scores, 75)),
    'max'     : float(all_scores.max()),
    'mean'    : float(all_scores.mean()),
    'std dev' : float(all_scores.std()),
}

print("Fisher‐Score Distribution Summary")
for k,v in stats.items():
    print(f"  {k:8s}: {v:.6e}")

# 4) Plot a histogram (log‐scaled x‐axis if scores span many orders)
plt.figure(figsize=(6,4))
plt.hist(all_scores, bins=100, log=True)
plt.xlabel("Fisher score")
plt.ylabel("Parameter count (log scale)")
plt.title("Histogram of Fisher scores")
plt.tight_layout()
plt.show()

# 5) (Optional) Per‐layer breakdown
for name, tensor in fisher.items():
    arr = tensor.cpu().numpy().flatten()
    print(f"{name:30s}  mean={arr.mean():.2e},  std={arr.std():.2e},  max={arr.max():.2e}")

