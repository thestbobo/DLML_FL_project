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
    Multi‐round layer‐wise TaLoS mask calibration on Q/K only.

    Args:
        model: torch.nn.Module (e.g. your DINO_ViT)
        fisher_scores: dict[str,Tensor]
           Precomputed Fisher diagonal for every parameter (all float Tensors).
        keep_ratio_per_block: float in (0,1], e.g. 0.10 for 10% keep at final.
        rounds: int ≥ 1
           Number of pruning iterations (R). Each iteration shrinks the alive set.

    Returns:
        masks: dict[str,Tensor]
           A dict mapping each param_name → FloatTensor mask (0.0 or 1.0).
           Only Q/K params may have 1.0; all others will be 0.0.
    """
    # 1) Initialize: for every param, start with mask = all zeros (float)
    masks = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # 2) Find all block indices by scanning for "blocks.{i}.attn.q_proj.weight"
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
    block_indices = sorted(block_indices)

    # 3) For each block i, we maintain a Boolean mask over that block’s Q/K
    for i in block_indices:
        q_w_name = f"blocks.{i}.attn.q_proj.weight"
        q_b_name = f"blocks.{i}.attn.q_proj.bias"
        k_w_name = f"blocks.{i}.attn.k_proj.weight"
        k_b_name = f"blocks.{i}.attn.k_proj.bias"

        # Check existence of biases
        has_qb = (q_b_name in fisher_scores)
        has_kb = (k_b_name in fisher_scores)

        # 3.1) Extract the Fisher scores for Q/K parameters in this block
        q_w_scores = fisher_scores[q_w_name]            # shape [D_q, D_model]
        k_w_scores = fisher_scores[k_w_name]            # shape [D_k, D_model]
        q_b_scores = fisher_scores[q_b_name] if has_qb else None
        k_b_scores = fisher_scores[k_b_name] if has_kb else None

        # 3.2) Flatten all Q/K scores into a single 1D vector
        parts_flat = [q_w_scores.reshape(-1), k_w_scores.reshape(-1)]
        if has_qb:
            parts_flat.append(q_b_scores.reshape(-1))
        if has_kb:
            parts_flat.append(k_b_scores.reshape(-1))

        all_scores_block = torch.cat(parts_flat, dim=0)  # length N_block
        N_block = all_scores_block.numel()
        if N_block == 0:
            # No Q/K in this block? (unlikely) → skip
            continue

        # 3.3) Initialize this block’s Boolean “alive” mask = all True
        #      We store it as a 1D BoolTensor of length N_block for convenience,
        #      then reshape back to each param’s shape each iteration.
        alive_bool = torch.ones(N_block, dtype=torch.bool, device=all_scores_block.device)

        # 3.4) Now iterate over r = 1..rounds to refine this block’s mask
        for r in range(1, rounds + 1):
            # (a) Desired keep‐fraction at iteration r:
            #     keep_frac = (keep_ratio_per_block)^(r / rounds)
            block_keep_frac = keep_ratio_per_block ** (r / float(rounds))

            # (b) Count how many are currently alive
            alive_indices = alive_bool.nonzero(as_tuple=False).reshape(-1)  # 1D indices
            num_alive = alive_indices.numel()
            if num_alive == 0:
                # nothing left alive → all zeros, break out
                alive_bool.zero_()
                break

            # (c) Compute how many to keep this round
            keep_n = int(math.ceil(block_keep_frac * num_alive))
            if keep_n < 1:
                keep_n = 1
            if keep_n >= num_alive:
                # keep everyone currently alive → skip
                continue

            # (d) Gather the Fisher scores of the currently‐alive subset
            #     “alive_indices” indexes into all_scores_block
            alive_scores = all_scores_block[alive_indices]

            # (e) Find threshold τ = smallest value among top-keep_n of alive_scores
            topk_vals, _ = torch.topk(alive_scores, keep_n, largest=True)
            tau = topk_vals.min()

            # (f) Update alive_bool: among those currently alive, only keep if score ≥ τ
            #    We do this by:
            #      alive_bool_new = (all_scores_block >= τ)  AND  alive_bool_old
            #    so that we only prune within “alive” set, not resurrect pruned ones.
            mask_this_round = (all_scores_block >= tau) & alive_bool
            alive_bool.copy_(mask_this_round)

            # loop to next r

        # 3.5) After R rounds, alive_bool marks exactly the top‐k entries (k ≈ k*N_block).
        #      Now we split alive_bool back into each of the four param‐tensors:
        #      - q_w_scores.reshape(-1) → shape = q_w_scores.numel()
        #      - k_w_scores.reshape(-1) → shape = k_w_scores.numel()
        #      - q_b_scores.reshape(-1) → if exists
        #      - k_b_scores.reshape(-1) → if exists

        # Start index pointer over alive_bool
        idx0 = 0

        #  • Q weight portion
        q_w_numel = q_w_scores.numel()
        q_w_alive = alive_bool[idx0 : idx0 + q_w_numel].reshape(q_w_scores.shape)
        masks[q_w_name].copy_(q_w_alive.float())  # Convert to float 0.0/1.0
        idx0 += q_w_numel

        #  • K weight portion
        k_w_numel = k_w_scores.numel()
        k_w_alive = alive_bool[idx0 : idx0 + k_w_numel].reshape(k_w_scores.shape)
        masks[k_w_name].copy_(k_w_alive.float())
        idx0 += k_w_numel

        #  • Q bias portion (if present)
        if has_qb:
            q_b_numel = q_b_scores.numel()
            q_b_alive = alive_bool[idx0 : idx0 + q_b_numel].reshape(q_b_scores.shape)
            masks[q_b_name].copy_(q_b_alive.float())
            idx0 += q_b_numel

        #  • K bias portion (if present)
        if has_kb:
            k_b_numel = k_b_scores.numel()
            k_b_alive = alive_bool[idx0 : idx0 + k_b_numel].reshape(k_b_scores.shape)
            masks[k_b_name].copy_(k_b_alive.float())
            idx0 += k_b_numel

        # At this point, idx0 == N_block. Move on to next block.

    return masks
