import torch
import warnings
import numpy as np

from collections import OrderedDict
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.exceptions import ConvergenceWarning

def svcca_score(X, Y, k=20, pca_dim=50, max_samples=2000):
    nX, nY = X.shape[0], Y.shape[0]
    if nY > nX:
        idx = np.random.choice(nY, size=nX, replace=False)
        Y = Y[idx]
    else:
        idx = np.random.choice(nX, size=nY, replace=False)
        X = X[idx]

    n = X.shape[0]
    if max_samples and n > max_samples:
        sel = np.random.choice(n, size=max_samples, replace=False)
        X = X[sel]
        Y = Y[sel]

    p = min(pca_dim, X.shape[1], Y.shape[1])
    pcaX = PCA(n_components=p).fit_transform(X)
    pcaY = PCA(n_components=p).fit_transform(Y)

    comp = min(k, p)
    cca = CCA(n_components=comp, max_iter=2000, tol=1e-4)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        Xc, Yc = cca.fit_transform(pcaX, pcaY)

    corrs = [np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(comp)]
    return float(np.mean(corrs))


def FedAlignAvg(
    weights_list: list[OrderedDict], # list of client model state_dicts
    repr_list: list[torch.Tensor],   # list of client representation tensors (n_i × D)
    num_samples_list: list[int],     # list of client sample counts n_i
    global_repr: torch.Tensor,       # server’s representation tensor (N × D)
    k: int = 20,                     # max number of CCA components to keep
    pca_dim: int = 50,               # number of PCA dimensions for preprocessing
    max_samples: int = 2000,         # upper bound on rows fed into SVCCA
) -> OrderedDict:

    """
    Drift-aware FedAvg using your svcca_score.

    Args:
        weights_list: List of client state_dicts (OrderedDict).
        repr_list:    List of client representation tensors (shape [n_i, D]).
        num_samples_list: List of sample counts n_i.
        global_repr:  Server’s reference repr tensor (shape [N, D]).
        k, pca_dim, max_samples: passed through to svcca_score.

    Returns:
        OrderedDict of aggregated model weights.
    """

    K = len(weights_list)
    if not (K == len(repr_list) == len(num_samples_list)):
        raise ValueError("weights_list, repr_list, and num_samples_list must all be the same length")

    G_np = global_repr.cpu().numpy()

    sim_scores = []
    for H in repr_list:
        H_np = H.cpu().numpy()
        sim = svcca_score(H_np, G_np, k=k, pca_dim=pca_dim, max_samples=max_samples)
        sim_scores.append(sim)

    raw = [s * n for s, n in zip(sim_scores, num_samples_list)]
    total = sum(raw) or 1e-12
    alphas = [r / total for r in raw]

    avg_weights = OrderedDict()
    for key in weights_list[0].keys():
        agg = torch.zeros_like(weights_list[0][key])
        for client_w, α in zip(weights_list, alphas):
            agg += client_w[key] * α
        avg_weights[key] = agg

    return avg_weights
