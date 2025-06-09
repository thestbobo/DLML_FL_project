def calibrate_mask_layerwise_qk_homemade(
    model,
    fisher_scores: dict,
    keep_ratio_per_block: float = 0.10,
    random_fallback_frac: float = 0.10,
    max_rounds: int = 5,
):
    """
    Build per-block masks that prune away the MOST-SENSITIVE Q/K rows (highest–Fisher)
    until we keep (1 - target_qk_sparsity) fraction of Q/K entries.  V‐rows are always kept.

    Args:
      - model:            a DINO_ViT instance (so keys look like "model.blocks.{i}.attn.qkv.weight")
      - fisher_scores:    dict[name→Tensor], the Fisher diagonal for every trainable param
      - keep_ratio_per_block: float in (0,1], e.g. 0.10 means “in each sub‐round we keep 10%
                              of currently‐alive Q/K entries” (used only if you want multi‐round logic).
      - random_fallback_frac: float in (0,1], if all Fisher scores in a block are zero, we
                              randomly keep this fraction of Q/K entries.
      - target_qk_sparsity: float in [0,1), fraction of Q/K entries to prune overall per block.
                           (e.g. 0.90 means “end up dropping 90% of all Q/K rows”)
      - max_rounds:       int ≥ 1, safety cap on how many iterative pruning rounds to do per block.

    Returns:
      masks: dict[name→FloatTensor(0.0/1.0)] containing exactly two keys per block:
             - "model.blocks.{i}.attn.qkv.weight"
             - "model.blocks.{i}.attn.qkv.bias" (if bias exists).
             Everything else in the model is untouched (not in `masks`).
    """
    target_qk_sparsity: float = 0.90
    masks = {}  # final binary masks (float of 0/1)

    # 1) collect all block indices that have a fused QKV layer
    block_indices = set()
    for name in fisher_scores:
        if name.startswith("model.blocks.") and name.endswith(".attn.qkv.weight"):
            parts = name.split(".")
            try:
                idx = int(parts[2])
                block_indices.add(idx)
            except ValueError:
                pass

    block_indices = sorted(block_indices)
    if not block_indices:
        # nothing to prune
        return masks

    # We'll do deterministic random fallback if needed.
    # But we need a generator on CUDA the same device as our fisher scores.
    # We will re-create it per‐block once we know what device the fisher tensor lives on.
    # (No need to reuse exact same RNG state across blocks; seeds are fixed per‐block.)

    for i in block_indices:
        w_name = f"model.blocks.{i}.attn.qkv.weight"
        b_name = f"model.blocks.{i}.attn.qkv.bias"

        # Retrieve Fisher diagonals
        base_w = fisher_scores[w_name]    # shape [3*D, D]
        device = base_w.device

        has_bias = (b_name in fisher_scores)
        if has_bias:
            base_b = fisher_scores[b_name]  # shape [3*D]

        # initialize masks (zeros, same shape)
        masks[w_name] = torch.zeros_like(base_w, dtype=torch.float32, device=device)
        if has_bias:
            masks[b_name] = torch.zeros_like(base_b, dtype=torch.float32, device=device)

        D_out, D_in = base_w.shape
        if D_out % 3 != 0:
            raise ValueError(f"Expected {w_name}.shape[0] divisible by 3, got {D_out}")
        chunk = D_out // 3  # rows per Q/K/V block

        # Flatten Q‐rows and K‐rows in both weight and bias
        q_w_flat = base_w[0:chunk, :].reshape(-1)
        k_w_flat = base_w[chunk:2*chunk, :].reshape(-1)

        if has_bias:
            qb_flat = base_b[0:chunk]
            kb_flat = base_b[chunk:2*chunk]
            all_scores = torch.cat([q_w_flat, k_w_flat, qb_flat, kb_flat], dim=0)
        else:
            all_scores = torch.cat([q_w_flat, k_w_flat], dim=0)

        N_qk = all_scores.numel()
        target_keep = max(1, int(math.ceil((1.0 - target_qk_sparsity) * N_qk)))
        # e.g. if target_sparsity=0.90, we want to KEEP 10% of N_qk entries overall.

        # If all Fisher scores are zero, we fallback to random
        if all_scores.max().item() == 0.0:
            # Randomly pick exactly target_keep indices to set alive
            rng = torch.Generator(device=device).manual_seed(0)
            perm = torch.randperm(N_qk, generator=rng, device=device)
            keep_idx = perm[:target_keep]

            # keep a fraction of entries at random
            fallback_n = max(1, int(math.ceil(random_fallback_frac * N_qk)))
            keep_idx = perm[:fallback_n]

            alive_mask = torch.zeros(N_qk, dtype=torch.bool, device=device)
            alive_mask[keep_idx] = True

        else:
            # iterative “TaLoS”‐style pruning rounds: progressively keep fewer
            # most‐sensitive entries until we've reduced to exactly target_keep.
            alive_mask = torch.ones(N_qk, dtype=torch.bool, device=device)

            # We'll do up to max_rounds; each round we keep only
            # keep_ratio_per_block fraction of currently‐alive entries, so that
            # across R rounds we end with (keep_ratio_per_block)^(R) * N_qk ≈ target_keep.
            # But since target_keep might not align exactly, we do this loop until
            # the number alive ≤ target_keep, then break and (if alive > target_keep) further
            # prune the surplus in one final pass.
            for round_idx in range(1, max_rounds+1):
                alive_indices = alive_mask.nonzero(as_tuple=False).reshape(-1)
                num_alive = alive_indices.numel()
                if num_alive <= target_keep:
                    break

                frac = keep_ratio_per_block ** (round_idx/float(max_rounds))
                # keep_frac = (Num_keep_probably ~ frac * num_alive)
                keep_n = max(1, int(math.floor(frac * num_alive)))
                current_scores = all_scores[alive_indices]

                topk_vals, topk_idxs = torch.topk(current_scores, keep_n, largest=True)

                tau = topk_vals.min()

                # keep entries with score ≥ tau
                new_alive = (all_scores <= tau) & alive_mask
                alive_mask = new_alive

            # After R rounds, we might have more than target_keep alive entries, or fewer.
            n_alive = alive_mask.sum().item()
            if n_alive > target_keep:
                # we need to drop (n_alive - target_keep) of the lowest‐scoring alive entries
                alive_indices = alive_mask.nonzero(as_tuple=False).reshape(-1)
                alive_scores = all_scores[alive_indices]
                # sort alive by descending score, keep the top target_keep
                top_vals, top_ids = torch.topk(alive_scores, target_keep, largest=True)
                to_keep = alive_indices[top_ids]  # global indices into all_scores
                new_alive = torch.zeros_like(alive_mask)
                new_alive[to_keep] = True
                alive_mask = new_alive
            elif n_alive < target_keep:
                # if we dropped too many, we randomly add back some of the previously‐dropped entries
                dropped = (~alive_mask).nonzero(as_tuple=False).reshape(-1)
                needed = target_keep - n_alive
                rng = torch.Generator(device=device).manual_seed(0)
                perm = torch.randperm(dropped.numel(), generator=rng, device=device)
                add_back = dropped[perm[:needed]]
                alive_mask[add_back] = True

        # Now alive_mask[j] is True iff that flattened Q/K entry should be kept.
        # We just need to write it back into the 2D mask for weight and bias.

        idx = 0
        # Q‐weights
        q_w_numel = chunk * D_in
        q_w_keep = alive_mask[idx: idx + q_w_numel].reshape((chunk, D_in))
        masks[w_name][0:chunk, :].copy_(q_w_keep.float())
        idx += q_w_numel

        # K‐weights
        k_w_numel = chunk * D_in
        k_w_keep = alive_mask[idx: idx + k_w_numel].reshape((chunk, D_in))
        masks[w_name][chunk:2*chunk, :].copy_(k_w_keep.float())
        idx += k_w_numel

        # Q‐bias and K‐bias if present
        if has_bias:
            q_b_numel = chunk
            q_b_keep = alive_mask[idx: idx + q_b_numel].reshape((chunk,))
            masks[b_name][0:chunk].copy_(q_b_keep.float())
            idx += q_b_numel

            k_b_numel = chunk
            k_b_keep = alive_mask[idx: idx + k_b_numel].reshape((chunk,))
            masks[b_name][chunk:2*chunk].copy_(k_b_keep.float())
            idx += k_b_numel

        # Finally, set ALL V‐rows (and V‐bias, if present) to 1.0:
        masks[w_name][2*chunk:3*chunk, :].fill_(1.0)
        if has_bias:
            masks[b_name][2*chunk:3*chunk].fill_(1.0)

    # (Optional) debug: count how many Q/K entries got kept vs. total
    total_keep = 0
    total_qk = 0
    for name, m in masks.items():
        if name.endswith(".attn.qkv.weight"):
            dout, din = m.shape
            bc = dout // 3
            keep_q = m[0:bc, :].sum().item()
            keep_k = m[bc:2*bc, :].sum().item()
            total_keep += keep_q + keep_k
            total_qk += (2 * bc * din)
        elif name.endswith(".attn.qkv.bias"):
            length = m.numel()
            bc = length // 3
            keep_qb = m[0:bc].sum().item()
            keep_kb = m[bc:2*bc].sum().item()
            total_keep += keep_qb + keep_kb
            total_qk += (2 * bc)

    if total_qk > 0:
        perc = 100.0 * total_keep / total_qk
        print(f">>> [DEBUG] FINAL Q/K KEEP = {int(total_keep)}/{int(total_qk)} ≈ {perc:.2f}%")
    else:
        print(">>> [DEBUG] No Q/K entries found in any block.")

    return masks