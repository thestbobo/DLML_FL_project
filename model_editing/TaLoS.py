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

# for centralized training
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

        available_scores = []
        for n in masks:
            tensor = fisher_scores[n]
            mask_bool = masks[n].bool()
            if mask_bool.any():
                available_scores.append(tensor[mask_bool].reshape(-1))
        if not available_scores:
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
            continue

        tau = torch.topk(all_scores, keep_n, largest=True).values.min()

        for n in masks:
            masks[n] = (fisher_scores[n] >= tau).float()

    return masks

# implemented for federated setting, globally prunes all parameters, this method is not as strategic as the QK, but allows us to prune more weights than QK
"""
implemented a mechanism to make sure non of the layers are pruned more than 95%,
it now bumps the global target of how many weights must survive in every layer if necessary,
resurrects weights when a layer dropped below the floor,
"""
def calibrate_mask_global(
    model: torch.nn.Module,
    calib_loader,
    criterion,
    device: torch.device,
    target_sparsity: float,
    rounds: int = 4,
    random_fallback_frac: float = 0.1,
    seed: int = 42,
    *,
    min_keep_frac: float = 0.05,
):
    """
    Multi-round global TaLoS calibration.

    `min_keep_frac`  :  hard safety floor per layer (0.05 = keep ≥5 %)
    `strict_final`   :  if True, re-drops the same number of params we had
                        to resurrect so the global keep count equals the
                        original `final_keep`.  Off by default.
    """
    model.to(device)

    orig = {n: p.data.clone().cpu() for n, p in model.named_parameters()}

    shapes = [(n, p.numel()) for n, p in model.named_parameters()]
    N = sum(sz for _, sz in shapes)
    alive = torch.ones(N, dtype=torch.bool, device=device)

    final_keep = max(1, math.ceil((1.0 - target_sparsity) * N))

    def is_whitelisted(name: str) -> bool:
        return (
            name.startswith("model.patch_embed")
            or name.startswith("model.pos_embed")
            or name.startswith("model.cls_token")
            or name.startswith("classifier")
        )

    layer_min_keep = []
    for name, sz in shapes:
        req = sz if is_whitelisted(name) else math.ceil(min_keep_frac * sz)
        layer_min_keep.append(req)
    min_total_keep = sum(layer_min_keep)
    if final_keep < min_total_keep:
        final_keep = min_total_keep

    for r in range(1, rounds + 1):
        ptr = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                cnt = param.numel()
                flat_orig = orig[name].view(-1).to(device)
                seg = alive[ptr : ptr + cnt].float().to(device)
                soft = seg + (1.0 - seg) * 0.1
                param.data.view(-1).copy_(flat_orig * soft)
                ptr += cnt

        fisher = compute_fisher_scores(model, calib_loader, criterion, device)

        ptr = 0
        for name, p in model.named_parameters():
            cnt = p.numel()
            if name in fisher:
                f = fisher[name].view(-1).to(device)
                f *= alive[ptr : ptr + cnt].float()
                fisher[name] = f.view_as(fisher[name])
            ptr += cnt

        all_scores = torch.cat([fisher[n].view(-1) for n, _ in shapes], 0)
        idxs = alive.nonzero(as_tuple=False).view(-1)
        cnt_alive = idxs.numel()

        keep_r = max(1, math.ceil((1 - target_sparsity) ** (r / rounds) * cnt_alive))
        sub = all_scores[idxs]

        if sub.max().item() == 0.0:
            rng = torch.Generator(device=device).manual_seed(seed + r)
            perm = torch.randperm(cnt_alive, generator=rng, device=device)
            keep_fb = max(1, math.ceil(random_fallback_frac * cnt_alive))
            new_alive = torch.zeros_like(alive)
            new_alive[idxs[perm[:keep_fb]]] = True
            alive = new_alive
        else:
            thr = torch.topk(sub, keep_r, largest=False).values.max()
            alive &= (all_scores <= thr)

        ptr = 0
        for name, sz in shapes:
            if is_whitelisted(name):
                alive[ptr : ptr + sz] = True
            ptr += sz

    all_scores = torch.cat([fisher[n].view(-1) for n, _ in shapes], 0)
    alive_idxs = alive.nonzero(as_tuple=False).view(-1)
    curr = alive_idxs.numel()

    if curr > final_keep:
        worst = torch.topk(all_scores[alive_idxs], curr - final_keep, largest=True).indices
        alive[alive_idxs[worst]] = False
    elif curr < final_keep:
        dropped = (~alive).nonzero(as_tuple=False).view(-1)
        need = final_keep - curr
        best = torch.topk(all_scores[dropped], need, largest=True).indices
        alive[dropped[best]] = True

    ptr = 0
    extra_alive = 0
    for (name, sz), min_k in zip(shapes, layer_min_keep):
        seg = alive[ptr : ptr + sz]
        kept = int(seg.sum())
        if kept < min_k:
            need = min_k - kept
            scores = all_scores[ptr : ptr + sz]
            dead_idx = (~seg).nonzero(as_tuple=False).view(-1)
            if dead_idx.numel():
                top = torch.topk(scores[dead_idx], need, largest=True).indices
                seg[dead_idx[top]] = True
                extra_alive += need
        ptr += sz

    masks, ptr = {}, 0
    params = dict(model.named_parameters())
    for name, sz in shapes:
        bin_mask = alive[ptr : ptr + sz].float().to(device)
        masks[name] = bin_mask.view_as(params[name])
        ptr += sz

    kept = int(alive.sum())
    print(f"[MASK SUMMARY] Kept {kept}/{N} params "
          f"({100*kept/N:.1f}% vs. target {(1-target_sparsity)*100:.1f}%)")
    print(f"[DEBUG] min_keep_frac = {min_keep_frac*100:.1f}% safety floor enforced")
    print("[MASK DIAG] per-layer keep%:")
    for n, m in masks.items():
        pct = 100 * m.sum().item() / m.numel()
        print(f"  {n:50s} -> {pct:5.1f}%")

    return masks

# qk layer pruning (prunes the most sensitive weights)
def calibrate_mask_layerwise_qk(
    model,
    fisher_scores: dict,
    target_qk_sparsity: float,
    keep_ratio_per_block: float = 0.10,
    random_fallback_frac: float = 0.10,
    max_rounds: int = 5,
):
    """
    Prune exactly `target_qk_sparsity` fraction of Q/K rows (V always kept),
    using an R-round iterative schedule controlled by keep_ratio_per_block.
    """
    masks = {}
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

        masks[w_name] = torch.zeros_like(W, device=device)
        if has_bias:
            masks[b_name] = torch.zeros_like(B, device=device)

        D_out, D_in = W.shape
        chunk = D_out // 3
        weight_block_size = chunk * D_in

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

        if all_scores.max().item() == 0.0:
            rng = torch.Generator(device=device).manual_seed(0)
            perm = torch.randperm(N_qk, generator=rng, device=device)
            keep_n = max(1, math.ceil(random_fallback_frac * N_qk))
            alive = torch.zeros(N_qk, dtype=torch.bool, device=device)
            alive[perm[:keep_n]] = True
        else:
            alive = torch.ones(N_qk, dtype=torch.bool, device=device)
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


        ptr = 0
        w_flat = masks[w_name].view(-1)
        w_flat[ptr : ptr + weight_block_size] = alive[ptr : ptr + weight_block_size].float()
        ptr += weight_block_size
        w_flat[ptr : ptr + weight_block_size] = alive[ptr : ptr + weight_block_size].float()
        masks[w_name][2*chunk : 3*chunk].fill_(1.0)

        if has_bias:
            bias_offset = 2 * weight_block_size
            qb_alive = alive[bias_offset : bias_offset + chunk]
            kb_alive = alive[bias_offset + chunk : bias_offset + 2*chunk]
            masks[b_name][0:chunk] = qb_alive.float()
            masks[b_name][chunk:2*chunk] = kb_alive.float()
            masks[b_name][2*chunk:3*chunk].fill_(1.0)

    return masks


# qk layer pruning (prunes the least sensitive weights)
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

    masks = {}

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

        return masks

    for i in block_indices:
        w_name = f"model.blocks.{i}.attn.qkv.weight"
        b_name = f"model.blocks.{i}.attn.qkv.bias"

        base_w = fisher_scores[w_name]
        device = base_w.device

        has_bias = (b_name in fisher_scores)
        if has_bias:
            base_b = fisher_scores[b_name]

        masks[w_name] = torch.zeros_like(base_w, dtype=torch.float32, device=device)
        if has_bias:
            masks[b_name] = torch.zeros_like(base_b, dtype=torch.float32, device=device)

        D_out, D_in = base_w.shape
        if D_out % 3 != 0:
            raise ValueError(f"Expected {w_name}.shape[0] divisible by 3, got {D_out}")
        chunk = D_out // 3

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

        if all_scores.max().item() == 0.0:
            rng = torch.Generator(device=device).manual_seed(0)
            perm = torch.randperm(N_qk, generator=rng, device=device)

            fallback_n = max(1, int(math.ceil(random_fallback_frac * N_qk)))
            keep_idx = perm[:fallback_n]

            alive_mask = torch.zeros(N_qk, dtype=torch.bool, device=device)
            alive_mask[keep_idx] = True

        else:
            alive_mask = torch.ones(N_qk, dtype=torch.bool, device=device)
            for round_idx in range(1, max_rounds+1):
                alive_indices = alive_mask.nonzero(as_tuple=False).reshape(-1)
                num_alive = alive_indices.numel()
                if num_alive <= target_keep:
                    break

                frac = keep_ratio_per_block ** (round_idx/float(max_rounds))
                keep_n = max(1, int(math.ceil(frac * num_alive)))
                current_scores = all_scores[alive_indices]

                topk_vals, topk_idxs = torch.topk(current_scores, keep_n, largest=False)

                tau = topk_vals.min()

                new_alive = (all_scores >= tau) & alive_mask
                alive_mask = new_alive

            n_alive = alive_mask.sum().item()
            if n_alive > target_keep:
                alive_indices = alive_mask.nonzero(as_tuple=False).reshape(-1)
                alive_scores = all_scores[alive_indices]

                top_vals, top_ids = torch.topk(alive_scores, target_keep, largest=True)
                to_keep = alive_indices[top_ids]

                new_alive = torch.zeros_like(alive_mask)
                new_alive[to_keep] = True

                alive_mask = new_alive
            elif n_alive < target_keep:
                dropped = (~alive_mask).nonzero(as_tuple=False).reshape(-1)
                needed = target_keep - n_alive

                rng = torch.Generator(device=device).manual_seed(0)
                perm = torch.randperm(dropped.numel(), generator=rng, device=device)
                add_back = dropped[perm[:needed]]

                alive_mask[add_back] = True


        idx = 0
        q_w_numel = chunk * D_in
        q_w_keep = alive_mask[idx: idx + q_w_numel].reshape((chunk, D_in))
        masks[w_name][0:chunk, :].copy_(q_w_keep.float())
        idx += q_w_numel

        k_w_numel = chunk * D_in
        k_w_keep = alive_mask[idx: idx + k_w_numel].reshape((chunk, D_in))
        masks[w_name][chunk:2*chunk, :].copy_(k_w_keep.float())
        idx += k_w_numel

        if has_bias:
            q_b_numel = chunk
            q_b_keep = alive_mask[idx: idx + q_b_numel].reshape((chunk,))
            masks[b_name][0:chunk].copy_(q_b_keep.float())
            idx += q_b_numel

            k_b_numel = chunk
            k_b_keep = alive_mask[idx: idx + k_b_numel].reshape((chunk,))
            masks[b_name][chunk:2*chunk].copy_(k_b_keep.float())
            idx += k_b_numel

        masks[w_name][2*chunk:3*chunk, :].fill_(1.0)
        if has_bias:
            masks[b_name][2*chunk:3*chunk].fill_(1.0)

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
