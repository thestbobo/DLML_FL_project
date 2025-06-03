import math

import torch


def compute_fisher_scores(model, dataloader, criterion, device):
    model.eval()
    fisher_scores = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_scores[name] += param.grad.pow(2)

    n_batches = len(dataloader)
    if n_batches > 0:
        for name in fisher_scores:
            fisher_scores[name] /= float(n_batches)

    return fisher_scores


def calibrate_mask(fisher_scores, target_sparsity, rounds):
    """
    Calibrate binary masks over multiple 'rounds' to reach target_sparsity.
    At each round r, we keep the top (1 – s)^(r/R) fraction of weights
    (i.e. highest‐sensitivity), pruning the rest.
    """
    masks = {n: torch.ones_like(s) for n, s in fisher_scores.items()}
    total = sum(m.numel() for m in masks.values())

    for r in range(1, rounds + 1):
        keep_frac = (1 - target_sparsity) ** (r / rounds)
        keep_n    = int(total * keep_frac)

        # Gather all currently‐alive scores
        available_scores = []
        for n in masks:
            tensor    = fisher_scores[n]
            mask_bool = masks[n].bool()
            if mask_bool.any():
                available_scores.append(tensor[mask_bool].reshape(-1))
        if not available_scores:
            # Nothing left to keep
            for n in masks:
                masks[n] = torch.zeros_like(masks[n])
            break

        all_scores = torch.cat(available_scores)
        num_avail  = all_scores.numel()

        if keep_n <= 0:
            for n in masks:
                masks[n] = torch.zeros_like(masks[n])
            break
        if keep_n >= num_avail:
            # Keep everything currently alive; no pruning this round.
            continue

        # Find the threshold τ = the smallest value among the top‐keep_n scores
        tau = torch.topk(all_scores, keep_n, largest=True).values.min()

        # ─── KEEP the weights whose fisher_score >= τ (i.e. the top keep_n) ───
        for n in masks:
            masks[n] = (fisher_scores[n] >= tau).float()

    return masks


# one-shot pruning (fixed threshold) -> not used right now, it's noisy
def calibrate_mask_one_shot(model, fisher_scores, target_sparsity, rounds):
    masks = {name: torch.ones_like(param) for name, param in model.named_parameters() if param.requires_grad}
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    target_params = int(total_params * (1 - target_sparsity))       # params to be kept

    for _ in range(rounds):
        # Flatten scores and masks
        all_scores = torch.cat([fisher_scores[name][masks[name] > 0].flatten() for name in fisher_scores])
        threshold = torch.topk(all_scores, target_params, largest=False).values[-1]

        # Update masks
        for name in masks:
            masks[name] = (fisher_scores[name] <= threshold).float()

        # Update target params for next round
        target_params = int(target_params * (1 - target_sparsity / rounds))

    return masks


def calibrate_mask_layerwise_qk(model, fisher_scores, keep_ratio_per_block=0.10):
    """
    Build a *float* mask (same dtype as model parameters) that, for each Transformer block i,
    keeps only the top `keep_ratio_per_block` fraction of Q/K parameters (weights + biases).
    All other parameters (MLP, LayerNorm, etc.) are masked to 0.

    We never call .float(), .to(), or .type()—instead, boolean masks get multiplied by
    `torch.ones_like(...)` to produce float masks.

    Args:
        model: torch.nn.Module         (e.g. your DINO_ViT)
        fisher_scores: dict[str, Tensor]
            A dict of {param_name: fisher_score_tensor} (all float).
        keep_ratio_per_block: float in (0,1], e.g. 0.10 for 10%.

    Returns:
        masks: dict[str, Tensor]
            Each mask is the same shape and dtype as its parameter; entries ∈ {0.0, 1.0}.
            “1.0” means “keep/update,” “0.0” means “freeze.”
    """
    # 1) Initialize every mask to float zeros (same dtype as param)
    masks = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # 2) Identify block indices from parameter names
    block_indices = set()
    for name in fisher_scores:
        if ".attn.q_proj.weight" in name:
            parts = name.split('.')
            if parts[0] == "blocks":
                try:
                    idx = int(parts[1])
                    block_indices.add(idx)
                except:
                    pass

    block_indices = sorted(block_indices)  # e.g. [0,1,2,...]

    # 3) For each block i, build a float mask on Q/K parameters
    for i in block_indices:
        q_w_name = f"blocks.{i}.attn.q_proj.weight"
        q_b_name = f"blocks.{i}.attn.q_proj.bias"
        k_w_name = f"blocks.{i}.attn.k_proj.weight"
        k_b_name = f"blocks.{i}.attn.k_proj.bias"

        has_qb = (q_b_name in fisher_scores)
        has_kb = (k_b_name in fisher_scores)

        # 3.1) Extract Q/K Fisher scores for this block
        q_w_scores = fisher_scores[q_w_name]               # shape [D_q, D_model]
        k_w_scores = fisher_scores[k_w_name]               # shape [D_k, D_model]
        q_b_scores = fisher_scores[q_b_name] if has_qb else None
        k_b_scores = fisher_scores[k_b_name] if has_kb else None

        # 3.2) Flatten and concatenate all Q/K scores into one vector
        sub_scores = [q_w_scores.reshape(-1), k_w_scores.reshape(-1)]
        if has_qb:
            sub_scores.append(q_b_scores.reshape(-1))
        if has_kb:
            sub_scores.append(k_b_scores.reshape(-1))

        all_scores_block = torch.cat(sub_scores, dim=0)  # 1D tensor length N_block
        N_block = all_scores_block.numel()
        if N_block == 0:
            continue

        # 3.3) Determine how many to keep in this block
        keep_n = int(math.ceil(keep_ratio_per_block * N_block))
        if keep_n < 1:
            keep_n = 1
        if keep_n >= N_block:
            # Keep all Q/K parameters in this block (mask=1.0 everywhere)
            masks[q_w_name].copy_(torch.ones_like(q_w_scores))
            masks[k_w_name].copy_(torch.ones_like(k_w_scores))
            if has_qb:
                masks[q_b_name].copy_(torch.ones_like(q_b_scores))
            if has_kb:
                masks[k_b_name].copy_(torch.ones_like(k_b_scores))
            continue

        # 3.4) Find threshold τ: the smallest value among the top‐keep_n scores
        topk_vals, _ = torch.topk(all_scores_block, keep_n, largest=True)
        tau = topk_vals.min()

        # 3.5) Build boolean masks, then convert to float by multiplying by ones_like
        q_w_bool = (q_w_scores >= tau)    # BoolTensor
        k_w_bool = (k_w_scores >= tau)    # BoolTensor

        masks[q_w_name].copy_(q_w_bool * torch.ones_like(q_w_scores))
        masks[k_w_name].copy_(k_w_bool * torch.ones_like(k_w_scores))

        if has_qb:
            q_b_bool = (q_b_scores >= tau)
            masks[q_b_name].copy_(q_b_bool * torch.ones_like(q_b_scores))
        if has_kb:
            k_b_bool = (k_b_scores >= tau)
            masks[k_b_name].copy_(k_b_bool * torch.ones_like(k_b_scores))

    return masks
