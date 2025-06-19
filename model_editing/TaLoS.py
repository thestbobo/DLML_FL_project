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

# this was used in centralized, replaced with calibrate_mask
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


def calibrate_mask_global(
    model: torch.nn.Module,
    calib_loader,
    criterion,
    device: torch.device,
    target_sparsity: float,   # e.g. 0.90 → prune 90% of Q/K entries
    rounds: int = 4,
    random_fallback_frac: float = 0.1,
    seed: int = 42,
):
    """
    Multi-round global prune *only* on Q and K rows of each attn.qkv.*:
      - Recompute Fisher on soft-masked model each round
      - Geometric bottom-k prune of Q+K entries
      - Leave everything else at mask=1
    Returns masks: name → FloatTensor (1=keep, 0=prune)
    """
    # 1) identify all Q/K entries
    # we'll build a single flat index of length total_QK = sum over blocks of (2*chunk*D_in + 2*chunk)
    qkv_names = []
    param_list = list(model.named_parameters())
    for name, p in param_list:
        if name.endswith("attn.qkv.weight") or name.endswith("attn.qkv.bias"):
            qkv_names.append(name)
    if not qkv_names:
        raise RuntimeError("No attn.qkv.* found in model")

    # gather shapes and compute offsets into a single flat QK index
    offsets = {}    # name → (start, length)
    total = 0
    for name in qkv_names:
        sz = param_list[[n for n,_ in param_list].index(name)][1].numel()
        # for weight: D_out = 3*chunk, bias: same length 3*chunk
        # but we only prune first 2/3 of entries (Q+K)
        chunk = sz // 3
        # we'll interpret the first 2*chunk entries in the flattened tensor as Q+K
        prune_len = 2 * chunk
        offsets[name] = (total, prune_len)
        total += prune_len

    # alive_QK: which of those total entries are still alive
    alive = torch.ones(total, dtype=torch.bool, device=device)
    final_keep = max(1, math.ceil((1.0 - target_sparsity) * total))

    model.to(device)
    # stash original
    orig = {name: p.data.clone().to(device) for name, p in param_list}

    def apply_soft_mask():
        # copy back into model.params: orig * (alive?1:0.1) on Q/K rows, leave V rows unchanged
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in offsets:
                    flat = orig[name].view(-1)
                    start, keep_len = offsets[name]
                    soft = torch.ones_like(flat, device=device)
                    # Q/K part
                    seg = alive[start:start+keep_len].float().to(device)
                    soft_qk = seg + (1.0 - seg) * 0.1
                    soft[:keep_len] = soft_qk
                    # V rows (last third) stay 1.0
                    param.data.copy_(soft.view_as(param.data))
                else:
                    # leave everything else at its orig value
                    param.data.copy_(orig[name])

    # 2) multi-round prune on Q/K
    for r in range(rounds):
        apply_soft_mask()

        # recompute fisher on the entire model
        fisher = compute_fisher_scores(model, calib_loader, criterion, device)

        # build flat QK score vector
        all_scores = torch.zeros(total, device=device)
        for name in qkv_names:
            fs = fisher[name].view(-1)
            start, keep_len = offsets[name]
            all_scores[start:start+keep_len] = fs[:keep_len]

        # zero out already-dead
        all_scores *= alive.float()

        # bottom-k among alive
        idxs = alive.nonzero(as_tuple=False).view(-1)
        cnt_alive = idxs.numel()
        keep_r = max(1, math.ceil((1.0 - target_sparsity) ** ((r+1)/rounds) * cnt_alive))
        sub = all_scores[idxs]
        if sub.numel()==0 or sub.max().item()==0.0:
            rng = torch.Generator(device=device).manual_seed(seed+r)
            perm = torch.randperm(cnt_alive, generator=rng, device=device)
            fb = max(1, math.ceil(random_fallback_frac*cnt_alive))
            new = torch.zeros_like(alive)
            new[idxs[perm[:fb]]] = True
            alive = new
        else:
            vals,_ = torch.topk(sub, keep_r, largest=False)
            tau = vals.max()
            alive = alive & (all_scores <= tau)

    # 3) final exact adjust
    idxs = alive.nonzero(as_tuple=False).view(-1)
    curr = idxs.numel()
    if curr>final_keep:
        vals = all_scores[idxs]
        drop_n = curr-final_keep
        _, didx = torch.topk(vals, drop_n, largest=True)
        alive[idxs[didx]] = False
    elif curr<final_keep:
        dropped = (~alive).nonzero(as_tuple=False).view(-1)
        need = final_keep-curr
        vals = all_scores[dropped]
        _, kidx = torch.topk(vals, need, largest=False)
        alive[dropped[kidx]] = True

    # 4) build final masks: 1.0 everywhere; 0.0 on pruned Q/K entries
    masks = {}
    for name, param in model.named_parameters():
        if name in offsets:
            flat = torch.ones(param.numel(), device=device)
            start, keep_len = offsets[name]
            flat[:keep_len][~alive[start:start+keep_len]] = 0.0
            masks[name] = flat.view_as(param).clone()
        else:
            masks[name] = torch.ones_like(param)

    # debug
    print("[MASK DIAG QK-only]")
    for name in qkv_names:
        m = masks[name]
        pct = 100 * float(m.view(-1)[:offsets[name][1]].sum()) / offsets[name][1]
        print(f"{name:50s} → {pct:5.1f}% of Q/K kept")

    return masks




# this one should work the same as qk but prunes the least sensitive weights instead
def calibrate_mask_layerwise_qk_ls(
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

    return masks