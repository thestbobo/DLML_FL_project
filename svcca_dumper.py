import glob
import os
import warnings

import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.exceptions import ConvergenceWarning

""" takes local saved representations and calculates svcca, dumps the value to a csv file """

# Base directory for your .pt representation files. Change it if you are on the drive
BASE = r""

def load_reps(path_pattern, layer):
    files = glob.glob(path_pattern)
    print(f"[DEBUG] Looking for {path_pattern!r}: found {len(files)} files")
    reps = {}
    for fn in files:
        ck = torch.load(fn, map_location="cpu", weights_only=False)
        r = ck["round"]
        mat = ck["representations"][layer].cpu().numpy()
        # flatten to (n_samples, n_features)
        if mat.ndim > 2:
            mat = mat.reshape(mat.shape[0], -1)
        cid = ck.get("client_id", "global")
        reps.setdefault(r, {})[cid] = mat
    return reps

def svcca_score(X, Y, k=20, pca_dim=50, max_samples=2000):
    # Subsample global Y down to the same number of rows as client X
    nX, nY = X.shape[0], Y.shape[0]
    if nY > nX:
        idx = np.random.choice(nY, size=nX, replace=False)
        Y = Y[idx]
    else:
        idx = np.random.choice(nX, size=nY, replace=False)
        X = X[idx]

    n = X.shape[0]
    # (Optional) further subsample to at most max_samples
    if max_samples and n > max_samples:
        sel = np.random.choice(n, size=max_samples, replace=False)
        X = X[sel]
        Y = Y[sel]

    # PCA‐reduce to pca_dim
    p = min(pca_dim, X.shape[1], Y.shape[1])
    pcaX = PCA(n_components=p).fit_transform(X)
    pcaY = PCA(n_components=p).fit_transform(Y)

    # CCA with higher max_iter and looser tol
    comp = min(k, p)
    cca = CCA(n_components=comp, max_iter=2000, tol=1e-4)

    # suppress the ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        Xc, Yc = cca.fit_transform(pcaX, pcaY)

    corrs = [np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(comp)]
    return float(np.mean(corrs))

if __name__ == "__main__":
    layers = [
        "model.blocks.3.attn.qkv",
        "model.blocks.6.mlp.fc2",
        "model.blocks.11.attn.qkv",
    ]
    rows = []

    for layer in layers:
        for num_client in range(10):
            for rnd in range(5, 51, 5):
                client_pattern = os.path.join(
                    BASE, f"client{num_client}_round{rnd}.pt"
                )
                global_pattern = os.path.join(
                    BASE, f"clientglobal_round{rnd}.pt"
                )

                client_reps = load_reps(client_pattern, layer)
                global_reps = load_reps(global_pattern, layer)

                for rr, gd in global_reps.items():
                    G = gd["global"]
                    for cid, C in client_reps.get(rr, {}).items():
                        try:
                            print(f"[DEBUG] Client {cid} has {G.shape[0]} rows and {len(C)} components")
                            score = svcca_score(C, G,
                                                k=20,
                                                pca_dim=50,
                                                max_samples=2000)
                        except Exception as e:
                            print(f"[WARNING] Skipping layer={layer} "
                                  f"round={rr} client={cid} due to: {e}")
                            continue
                        rows.append({
                            "layer": layer,
                            "round": rr,
                            "client": cid,
                            "svcca": score
                        })

    df = pd.DataFrame(rows)
    df.to_csv("svcca_summary.csv", index=False)
    print("Wrote svcca_summary.csv with", len(df), "rows")
