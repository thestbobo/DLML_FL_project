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
        keep_n = int(total * keep_frac)

        # Gather all currently‐alive scores
        available_scores = []
        for n in masks:
            tensor = fisher_scores[n]
            mask_bool = masks[n].bool()
            if mask_bool.any():
                available_scores.append(tensor[mask_bool].reshape(-1))
        if not available_scores:
            # Nothing left to keep
            for n in masks:
                masks[n] = torch.zeros_like(masks[n])
            break

        all_scores = torch.cat(available_scores)
        num_avail = all_scores.numel()

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


def calibrate_mask_layerwise_qk(
    model,
    fisher_scores,
    keep_ratio_per_block=0.10,
    rounds=5
):
    """
    TaLoS mask calibration (layer‐wise) over Q/K only for DINO_ViT.

    – model: a DINO_ViT instance (so model.named_parameters() yields names like
             "model.blocks.0.attn.qkv.weight", etc.).
    – fisher_scores: dict[name→Tensor] from compute_fisher_scores(model,...).
    – keep_ratio_per_block: e.g. 0.10 means “at the final round, keep 10% of Q/K.”
      Internally, we do R rounds of (1 – s)^(r/R).
    – rounds: how many iterative prune‐refine steps to do.

    Returns: masks, a dict[name→FloatTensor of 0.0/1.0] for every trainable param.
             Only the Q‐ and K‐rows in each “model.blocks.{i}.attn.qkv” layer will have
             some 1.0s.  Everything else remains at 0.0.
    """

    # Initialize: for every param, start with mask = all zeros (float)
    masks = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Find all block indices by scanning for "model.blocks.{i}.attn.qkv.weight"
    block_indices = set()
    for name in fisher_scores:
        if "model.blocks," in name and name.endswith(".attn.qkv.weight"):
            parts = name.split('.')

            try:
                idx = int(parts[2])
                block_indices.add(idx)
            except:
                pass
    block_indices = sorted(block_indices)

    # For each block i, carve out a mask over only the Q/K rows of the fused qkv matrix
    for i in block_indices:
        # — Exact names for this block’s fused QKV weight & bias:
        qkv_w_name = f"model.blocks.{i}.attn.qkv.weight"
        qkv_b_name = f"model.blocks.{i}.attn.qkv.bias"

        if qkv_w_name not in fisher_scores:
            raise KeyError(f"Expected '{qkv_w_name}' in fisher_scores but it was not found.")

        has_bias = (qkv_b_name in fisher_scores)

        #  Load the fused QKV fisher‐diagonal
        qkv_w_scores = fisher_scores[qkv_w_name]  # Tensor of shape [3*D, D]
        qkv_b_scores = fisher_scores[qkv_b_name] if has_bias else None  # [3*D] or None

        # Split into Q vs. K portions (ignore V)
        D_out, D_in = qkv_w_scores.shape
        if D_out % 3 != 0:
            raise ValueError(
                f"Expected '{qkv_w_name}' to have out‐dim multiple of 3, but got {D_out}"
            )
        block_chunk = D_out // 3  # number of rows per Q, K, V slice

        # Flatten Q‐weight and K‐weight rows into 1D
        q_w_flat = qkv_w_scores[0: block_chunk, :].reshape(-1)
        k_w_flat = qkv_w_scores[block_chunk: 2 * block_chunk, :].reshape(-1)

        # Flatten Q‐bias and K‐bias if bias exists
        if has_bias:
            qb_flat = qkv_b_scores[0: block_chunk]
            kb_flat = qkv_b_scores[block_chunk: 2 * block_chunk]
            all_scores_block = torch.cat([q_w_flat, k_w_flat, qb_flat, kb_flat], dim=0)
        else:
            all_scores_block = torch.cat([q_w_flat, k_w_flat], dim=0)

        N_block = all_scores_block.numel()
        if N_block == 0:
            # No Q/K in this block → skip
            continue

        # Create a boolean “alive” mask (all True initially)
        alive_bool = torch.ones(N_block, dtype=torch.bool, device=all_scores_block.device)

        # Iterative prune‐and‐refine (R rounds)
        for r in range(1, rounds + 1):
            block_keep_frac = keep_ratio_per_block ** (r / float(rounds))
            alive_indices = alive_bool.nonzero(as_tuple=False).reshape(-1)
            num_alive = alive_indices.numel()
            if num_alive == 0:
                alive_bool.zero_()
                break

            keep_n = int(math.ceil(block_keep_frac * num_alive))
            if keep_n < 1:
                keep_n = 1
            if keep_n >= num_alive:
                # Keep all currently alive → no change
                continue

            alive_scores = all_scores_block[alive_indices]
            topk_vals, _ = torch.topk(alive_scores, keep_n, largest=True)
            tau = topk_vals.min()
            mask_this_round = (all_scores_block >= tau) & alive_bool
            alive_bool.copy_(mask_this_round)

        # 3.5) Now “alive_bool” marks exactly which Q/K entries survive. Copy back into masks.
        idx0 = 0

        # — Q‐weights slice:
        q_w_numel = block_chunk * D_in
        q_w_alive = alive_bool[idx0: idx0 + q_w_numel].reshape((block_chunk, D_in))
        masks[qkv_w_name][0: block_chunk, :].copy_(q_w_alive.float())
        idx0 += q_w_numel

        # — K‐weights slice:
        k_w_numel = block_chunk * D_in
        k_w_alive = alive_bool[idx0: idx0 + k_w_numel].reshape((block_chunk, D_in))
        masks[qkv_w_name][block_chunk: 2 * block_chunk, :].copy_(k_w_alive.float())
        idx0 += k_w_numel

        # — Q‐bias slice (if present):
        if has_bias:
            q_b_numel = block_chunk
            q_b_alive = alive_bool[idx0: idx0 + q_b_numel].reshape((block_chunk,))
            masks[qkv_b_name][0: block_chunk].copy_(q_b_alive.float())
            idx0 += q_b_numel

            # — K‐bias slice:
            k_b_numel = block_chunk
            k_b_alive = alive_bool[idx0: idx0 + k_b_numel].reshape((block_chunk,))
            masks[qkv_b_name][block_chunk: 2 * block_chunk].copy_(k_b_alive.float())
            idx0 += k_b_numel

        # The V‐portion (rows 2*block_chunk : 3*block_chunk) remains zero in both weight & bias

    return masks
