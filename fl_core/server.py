import torch
from collections import OrderedDict
from typing import List

def svcca_score(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int = 20,
    pca_dim: int = 50,
    max_samples: int = 2000,
    eps: float = 1e-10
) -> float:
    """
    Compute SVCCA score between X and Y using GPU-accelerated PyTorch.

    Args:
        X: Tensor of shape (n_x, d)
        Y: Tensor of shape (n_y, d)
        k: number of CCA components
        pca_dim: PCA dimensionality
        max_samples: max number of samples for SVCCA
        eps: regularization for covariance matrices

    Returns:
        Mean canonical correlation (float).
    """
    device = X.device
    Y = Y.to(device)

    # Subsample to match row counts
    nX, nY = X.size(0), Y.size(0)
    if nY > nX:
        idx = torch.randperm(nY, device=device)[:nX]
        Y = Y[idx]
    else:
        idx = torch.randperm(nX, device=device)[:nY]
        X = X[idx]

    # Further subsample if too large
    n = X.size(0)
    if max_samples and n > max_samples:
        sel = torch.randperm(n, device=device)[:max_samples]
        X = X[sel]
        Y = Y[sel]

    # Center data
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    # PCA reduction via torch.pca_lowrank
    p = min(pca_dim, Xc.size(1), Yc.size(1))
    Ux, Sx, Vx = torch.pca_lowrank(Xc, q=p)
    Uy, Sy, Vy = torch.pca_lowrank(Yc, q=p)
    Xp = Xc @ Vx[:, :p]
    Yp = Yc @ Vy[:, :p]

    # CCA via SVD
    comp = min(k, p)
    # Covariance estimates
    cov_xx = (Xp.T @ Xp) / (Xp.size(0) - 1) + eps * torch.eye(p, device=device)
    cov_yy = (Yp.T @ Yp) / (Yp.size(0) - 1) + eps * torch.eye(p, device=device)
    cov_xy = (Xp.T @ Yp) / (Xp.size(0) - 1)

    def inv_sqrt(mat: torch.Tensor) -> torch.Tensor:
        vals, vecs = torch.linalg.eigh(mat)
        inv_sqrt_vals = torch.diag(vals.clamp(min=eps).rsqrt())
        return vecs @ inv_sqrt_vals @ vecs.T

    inv_xx = inv_sqrt(cov_xx)
    inv_yy = inv_sqrt(cov_yy)

    M = inv_xx @ cov_xy @ inv_yy
    _, S_vals, _ = torch.linalg.svd(M)
    corrs = S_vals[:comp]
    return corrs.mean().item()


def FedAlignAvg(
    weights_list: List[OrderedDict],
    repr_list: List[torch.Tensor],
    num_samples_list: List[int],
    global_repr: torch.Tensor,
    k: int = 20,
    pca_dim: int = 50,
    max_samples: int = 2000
) -> OrderedDict:
    """
    GPU-aware FedAvg aggregated by SVCCA similarity.

    Args:
        weights_list: list of client state_dicts
        repr_list: list of client repr tensors (n_i × D)
        num_samples_list: list of sample counts
        global_repr: server repr tensor (N × D)
        k, pca_dim, max_samples: SVCCA params
    Returns:
        Aggregated state_dict
    """
    # Ensure device consistency
    device = global_repr.device
    G = global_repr.to(device)

    # Compute SVCCA similarities on GPU
    sim_scores = []
    for H in repr_list:
        H_dev = H.to(device)
        sim = svcca_score(H_dev, G, k=k, pca_dim=pca_dim, max_samples=max_samples)
        sim_scores.append(sim)

    # Raw weights ∝ sim_i * n_i
    raw = [s * n for s, n in zip(sim_scores, num_samples_list)]
    total = sum(raw) or 1e-12
    alphas = [r / total for r in raw]

    # Aggregate model weights (stay on CPU or original device)
    avg_weights = OrderedDict()
    for key in weights_list[0].keys():
        agg = torch.zeros_like(weights_list[0][key], device=weights_list[0][key].device)
        for client_w, alpha in zip(weights_list, alphas):
            agg = agg + client_w[key].to(agg.device) * alpha
        avg_weights[key] = agg

    return avg_weights
