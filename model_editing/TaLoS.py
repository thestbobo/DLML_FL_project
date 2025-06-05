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
    fisher_scores: dict,
    keep_ratio_per_block: float = 0.10,
    rounds: int = 5
):
    """
    TaLoS mask calibration for DINO_ViT fused QKV layers only.

    – We only insert mask entries for:
         "model.blocks.{i}.attn.qkv.weight"
         "model.blocks.{i}.attn.qkv.bias"

    – We prune Q and K (first 2/3 of each tensor) by Fisher‐score over `rounds`.
    – We force all V‐rows (last 1/3) to 1.0 (i.e. keep them).
    – All other parameters remain untouched (not in `masks`).

    Returns:
      masks: dict[name → FloatTensor(0/1)] that only contains QKV keys.
    """

    masks = {}  # only QKV keys will be inserted

    # Find which block indices contain a fused QKV layer
    block_indices = set()
    for name in fisher_scores:
        if name.startswith("model.blocks.") and name.endswith(".attn.qkv.weight"):
            parts = name.split(".")         # ["model","blocks","{i}","attn","qkv","weight"]
            try:
                idx = int(parts[2])
                block_indices.add(idx)
            except ValueError:
                pass

    block_indices = sorted(block_indices)
    print(">>> [DEBUG] block_indices found:", block_indices)
    if not block_indices:
        print("    [WARN] No QKV blocks matched; masks will stay empty.")

    # 2) For each block i, build a mask for Q+K (pruned) and V (kept)
    for i in block_indices:
        qkv_w_name = f"model.blocks.{i}.attn.qkv.weight"
        qkv_b_name = f"model.blocks.{i}.attn.qkv.bias"

        # Create zero‐filled mask tensors of the same shape as fisher_scores
        base_w = fisher_scores[qkv_w_name]  # shape [3*D, D]
        masks[qkv_w_name] = torch.zeros_like(base_w)

        has_bias = (qkv_b_name in fisher_scores)
        if has_bias:
            base_b = fisher_scores[qkv_b_name]  # shape [3*D]
            masks[qkv_b_name] = torch.zeros_like(base_b)

        # Extract Fisher‐scores for Q/K portions
        D_out, D_in = base_w.shape
        if D_out % 3 != 0:
            raise ValueError(f"Expected {qkv_w_name}.shape[0] divisible by 3, got {D_out}")
        block_chunk = D_out // 3  # # rows per Q/K/V

        # Flatten Q‐weights and K‐weights
        q_w_flat = base_w[0:block_chunk, :].reshape(-1)
        k_w_flat = base_w[block_chunk:2*block_chunk, :].reshape(-1)

        if has_bias:
            qb_flat = base_b[0:block_chunk]
            kb_flat = base_b[block_chunk:2*block_chunk]
            all_scores_block = torch.cat([q_w_flat, k_w_flat, qb_flat, kb_flat], dim=0)
        else:
            all_scores_block = torch.cat([q_w_flat, k_w_flat], dim=0)

        N_block = all_scores_block.numel()
        alive_bool = torch.ones(N_block, dtype=torch.bool, device=all_scores_block.device)

        # Iteratively prune Q/K over `rounds`
        for r in range(1, rounds + 1):
            block_keep_frac = keep_ratio_per_block ** (r / float(rounds))
            alive_indices = alive_bool.nonzero(as_tuple=False).reshape(-1)
            num_alive = alive_indices.numel()
            if num_alive == 0:
                alive_bool.zero_()
                break

            keep_n = max(1, int(math.ceil(block_keep_frac * num_alive)))
            # Find the threshold tau corresponding to the keep_n **lowest** Fisher scores
            alive_scores = all_scores_block[alive_indices]
            # To get lowest‐Fisher entries, do topk on -alive_scores
            neg_alive = -alive_scores
            topk_neg_vals, _ = torch.topk(neg_alive, keep_n, largest=True)
            tau_neg = topk_neg_vals.min()  # most negative among topk
            tau = -tau_neg  # corresponding positive threshold

            # Keep entries whose score <= tau (lowest Fisher), and were alive
            mask_this_round = (all_scores_block <= tau) & alive_bool
            alive_bool.copy_(mask_this_round)

        # 4) Write back Q/K “1.0” for the bottom (least sensitive) entries
        idx0 = 0

        q_w_numel = block_chunk * D_in
        q_w_alive = alive_bool[idx0: idx0 + q_w_numel].reshape((block_chunk, D_in))
        masks[qkv_w_name][0:block_chunk, :].copy_(q_w_alive.float())
        idx0 += q_w_numel

        k_w_numel = block_chunk * D_in
        k_w_alive = alive_bool[idx0: idx0 + k_w_numel].reshape((block_chunk, D_in))
        masks[qkv_w_name][block_chunk:2 * block_chunk, :].copy_(k_w_alive.float())
        idx0 += k_w_numel

        if has_bias:
            q_b_numel = block_chunk
            q_b_alive = alive_bool[idx0: idx0 + q_b_numel].reshape((block_chunk,))
            masks[qkv_b_name][0:block_chunk].copy_(q_b_alive.float())
            idx0 += q_b_numel

            k_b_numel = block_chunk
            k_b_alive = alive_bool[idx0: idx0 + k_b_numel].reshape((block_chunk,))
            masks[qkv_b_name][block_chunk:2 * block_chunk].copy_(k_b_alive.float())
            idx0 += k_b_numel

        # 5) Force all V‐rows (highest third) to 1 (keep)
        masks[qkv_w_name][2 * block_chunk: 3 * block_chunk, :].fill_(1.0)
        if has_bias:
            masks[qkv_b_name][2 * block_chunk: 3 * block_chunk].fill_(1.0)

    # 6) Debug: print final keep‐fraction of Q/K
    total_keep = 0
    total_qk = 0
    for name, m in masks.items():
        if name.endswith(".attn.qkv.weight"):
            D_out, D_in = m.shape
            bc = D_out // 3
            keep_q = m[0:bc, :].sum().item()
            keep_k = m[bc:2 * bc, :].sum().item()
            total_keep += (keep_q + keep_k)
            total_qk += (2 * bc * D_in)
        if name.endswith(".attn.qkv.bias"):
            length = m.numel()
            bc = length // 3
            keep_qb = m[0:bc].sum().item()
            keep_kb = m[bc:2 * bc].sum().item()
            total_keep += (keep_qb + keep_kb)
            total_qk += (2 * bc)

    if total_qk > 0:
        print(f">>> [DEBUG] FINAL MASK Q/K KEEP = {int(total_keep)}/{int(total_qk)} "
              f"≈ {100 * total_keep / total_qk:.2f}% (least sensitive kept)")
    else:
        print(">>> [DEBUG] FINAL MASK has no Q/K entries.")

    return masks