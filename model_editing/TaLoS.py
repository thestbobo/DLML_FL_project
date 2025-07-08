import math
from collections import OrderedDict

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

# this is used in centralized
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


# federated model, calibrates mask layerwise, priuning only Q/K layers untill reaching targert_qk_sparsity -> 90% QK pruned = 60% global weights pruned
def calibrate_mask_layerwise_qk(
    model,
    fisher_scores: dict,
    target_qk_sparsity: float,         # must now be passed in
    keep_ratio_per_block: float = 0.10,
    random_fallback_frac: float = 0.10,
    max_rounds: int = 5,
):
    """
    Prune exactly `target_qk_sparsity` fraction of Q/K rows (V always kept),
    using an R-round iterative schedule controlled by keep_ratio_per_block.
    """
    masks = {}
    # find all block indices
    blocks = sorted(
        int(n.split(".")[2])
        for n in fisher_scores
        if n.startswith("model.blocks.") and n.endswith(".attn.qkv.weight")
    )
    if not blocks:
        return masks

    for i in blocks:
        w_name = f"model.blocks.{i}.attn.qkv.weight"
        b_name = f"model.blocks.{i}.attn.qkv.bias"
        W = fisher_scores[w_name]
        device = W.device
        has_bias = (b_name in fisher_scores)
        B = fisher_scores[b_name] if has_bias else None

        # allocate zero masks
        masks[w_name] = torch.zeros_like(W, device=device)
        if has_bias:
            masks[b_name] = torch.zeros_like(B, device=device)

        D_out, D_in = W.shape
        chunk = D_out // 3
        weight_block_size = chunk * D_in

        # build flat score vector: [q_w, k_w, (q_b, k_b)]
        q_w = W[0:chunk].reshape(-1)
        k_w = W[chunk:2*chunk].reshape(-1)
        if has_bias:
            q_b = B[0:chunk]
            k_b = B[chunk:2*chunk]
            all_scores = torch.cat([q_w, k_w, q_b, k_b], dim=0)
        else:
            all_scores = torch.cat([q_w, k_w], dim=0)

        N_qk = all_scores.numel()
        target_keep = max(1, math.ceil((1.0 - target_qk_sparsity) * N_qk))

        # initialize alive mask
        if all_scores.max().item() == 0.0:
            rng = torch.Generator(device=device).manual_seed(0)
            perm = torch.randperm(N_qk, generator=rng, device=device)
            keep_n = max(1, math.ceil(random_fallback_frac * N_qk))
            alive = torch.zeros(N_qk, dtype=torch.bool, device=device)
            alive[perm[:keep_n]] = True
        else:
            alive = torch.ones(N_qk, dtype=torch.bool, device=device)
            # iterative bottom-k pruning
            for r in range(1, max_rounds + 1):
                idxs = alive.nonzero(as_tuple=False).view(-1)
                cnt = idxs.numel()
                if cnt <= target_keep:
                    break
                frac = keep_ratio_per_block ** (r / float(max_rounds))
                keep_n = max(1, math.ceil(frac * cnt))
                vals = all_scores[idxs]
                small_vals, _ = torch.topk(vals, keep_n, largest=False)
                tau = small_vals.max()
                alive &= (all_scores <= tau)

            # final adjust
            alive_idxs = alive.nonzero(as_tuple=False).view(-1)
            if alive_idxs.numel() > target_keep:
                vals = all_scores[alive_idxs]
                _, sel = torch.topk(vals, target_keep, largest=False)
                new_alive = torch.zeros_like(alive)
                new_alive[alive_idxs[sel]] = True
                alive = new_alive
            elif alive_idxs.numel() < target_keep:
                dropped = (~alive).nonzero(as_tuple=False).view(-1)
                needed = target_keep - alive_idxs.numel()
                rng = torch.Generator(device=device).manual_seed(0)
                perm = torch.randperm(dropped.numel(), generator=rng, device=device)
                alive[dropped[perm[:needed]]] = True

        # --- write back the masks ---

        # 1) weight: first chunk*D_in for Q, next chunk*D_in for K
        ptr = 0
        # Q weight rows
        w_flat = masks[w_name].view(-1)
        w_flat[ptr : ptr + weight_block_size] = alive[ptr : ptr + weight_block_size].float()
        ptr += weight_block_size
        # K weight rows
        w_flat[ptr : ptr + weight_block_size] = alive[ptr : ptr + weight_block_size].float()
        # V rows always kept
        masks[w_name][2*chunk : 3*chunk].fill_(1.0)

        # 2) bias, if present: bias is length D_out = 3*chunk
        if has_bias:
            # alive bias segments start at offset 2*weight_block_size
            bias_offset = 2 * weight_block_size
            # Q bias
            qb_alive = alive[bias_offset : bias_offset + chunk]
            # K bias
            kb_alive = alive[bias_offset + chunk : bias_offset + 2*chunk]
            # assign
            masks[b_name][0:chunk] = qb_alive.float()
            masks[b_name][chunk:2*chunk] = kb_alive.float()
            # V bias always kept
            masks[b_name][2*chunk:3*chunk].fill_(1.0)

    return masks


# implemented for federated setting, globally prunes all parameters, this method is not as strategic as the QK, but allows us to prune more weights than QK

"""
implemented a mechanism to make sure non of the layers are pruned more than 95%,
it now bumps the global target of how many weights must survive in every layer if necessary,
resurrects weights when a layer dropped below the floor,
if strict_final=True, drops the same count elsewhere without violating any layer’s floor
"""

def calibrate_mask_global(
    model, calib_loader, criterion, device,
    target_sparsity, rounds=4, random_fallback_frac=0.1,
    seed=42, min_keep_frac=0.05, mask_type="least_sensitive",
    strict_final=False
):
    model.to(device)
    # 1) Collect and flatten parameters
    orig_weights = []
    slices = []
    whitelisted = []
    offset = 0
    for name, p in model.named_parameters():
        numel = p.numel()
        orig_weights.append(p.data.detach().clone())
        slices.append((name, offset, offset + numel))
        whitelisted.append(name.startswith("model.patch_embed")
                           or name.startswith("model.pos_embed")
                           or name.startswith("model.cls_token")
                           or name.startswith("classifier"))
        offset += numel
    flat_orig = torch.cat([w.view(-1) for w in orig_weights], 0).to(device)
    N = flat_orig.numel()

    # 2) Build per-slice min‐keep
    layer_min_keep = torch.tensor([
        s_end - s_begin if wl else math.ceil(min_keep_frac * (s_end - s_begin))
        for (_, s_begin, s_end), wl in zip(slices, whitelisted)
    ], device=device)
    total_min_keep = int(layer_min_keep.sum())
    final_keep = max(math.ceil((1 - target_sparsity) * N), total_min_keep)

    # 3) Initialize alive mask
    alive = torch.ones(N, dtype=torch.bool, device=device)

    # 4) Masking rounds
    for r in range(1, rounds + 1):
        # restore alive weights
        model_weights = flat_orig * alive.float()
        # compute fisher or zeros
        if mask_type in ("least_sensitive","most_sensitive"):
            fisher_scores = compute_fisher_scores(
                model, calib_loader, criterion, device, N
            )
        else:
            fisher_scores = torch.zeros_like(flat_orig)
        # apply alive mask to scores
        scores = fisher_scores * alive.float()

        # transform scores
        if mask_type == "low_magnitude":
            scores = flat_orig.abs()
        elif mask_type == "high_magnitude":
            scores = flat_orig.abs()
        elif mask_type == "random":
            scores = torch.rand_like(flat_orig)
        # otherwise keep the fisher‐based ones

        # pick keep_r
        idxs = alive.nonzero(as_tuple=False).view(-1)
        keep_r = max(1, math.ceil((1 - target_sparsity)**(r/rounds) * idxs.numel()))
        sub_scores = scores[idxs]
        if sub_scores.max()==0 and sub_scores.min()==0:
            g = torch.Generator(device=device).manual_seed(seed + r)
            perm = torch.randperm(idxs.numel(), generator=g, device=device)
            fb = max(1, math.ceil(random_fallback_frac * idxs.numel()))
            new_alive = torch.zeros_like(alive)
            new_alive[idxs[perm[:fb]]] = True
        else:
            largest = mask_type in ("most_sensitive","high_magnitude")
            topk = torch.topk(sub_scores, keep_r, largest=largest)
            new_alive = torch.zeros_like(alive)
            new_alive[idxs[topk.indices]] = True

        # re‐whitelist
        for (name, b, e), wl in zip(slices, whitelisted):
            if wl:
                new_alive[b:e] = True
        alive = new_alive

    # 5) Final adjustments for exact sparsity
    keep = alive.sum().item()
    if strict_final and keep != final_keep:
        diff = keep - final_keep
        scores = scores  # reuse from last round
        if diff > 0:
            # drop worst among alive
            idxs = alive.nonzero().view(-1)
            worst = torch.topk(scores[idxs], diff, largest=False).indices
            alive[idxs[worst]] = False
        else:
            # add best among dead
            idxs = (~alive).nonzero().view(-1)
            best = torch.topk(scores[idxs], -diff, largest=True).indices
            alive[idxs[best]] = True
    elif keep < final_keep:
        # always bump up if under
        idxs = (~alive).nonzero().view(-1)
        need = final_keep - keep
        best = torch.topk(scores[idxs], need, largest=True).indices
        alive[idxs[best]] = True

    # 6) Enforce per‐layer minima
    for (name, b, e), min_k in zip(slices, layer_min_keep):
        seg = alive[b:e]
        if seg.sum() < min_k:
            dead = (~seg).nonzero().view(-1)
            to_add = torch.topk(scores[b:e][dead], int(min_k-seg.sum()), largest=True).indices
            alive[b+dead[to_add]] = True

    # 7) Unflatten into mask dict
    masks = {}
    for (name, b, e), w in zip(slices, orig_weights):
        masks[name] = alive[b:e].float().view_as(w).to(device)
    return masks



# this one should work the same as qk but prunes the least sensitive weights instead
def calibrate_mask_layerwise_qk_ls(
    model,
    fisher_scores: dict,
    target_qk_sparsity: float = 0.90,
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
                keep_n = max(1, int(math.ceil(frac * num_alive)))
                current_scores = all_scores[alive_indices]

                topk_vals, topk_idxs = torch.topk(current_scores, keep_n, largest=True)

                topk_vals, topk_idxs = torch.topk(current_scores, keep_n, largest=False)

                tau = topk_vals.min()

                # keep entries with score ≥ tau
                new_alive = (all_scores >= tau) & alive_mask
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

    print("[LAYER MASK DIAG] per-layer keep%:")
    for name, m in masks.items():
        kept = m.sum().item()
        total = m.numel()
        pct = 100.0 * kept / total
        print(f"  {name:50s} -> {pct:5.1f}% ({int(kept)}/{total})")

    return masks
