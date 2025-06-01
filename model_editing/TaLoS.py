import torch


def compute_fisher_scores(model, dataloader, criterion, device):
    """
    Compute Fisher Information matrix diagonal elements (sensitivity scores).
    """
    model.eval()
    fisher_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_scores[name] += param.grad ** 2

    # Normalize scores
    for name in fisher_scores:
        fisher_scores[name] /= len(dataloader)

    return fisher_scores


# iterative pruning, moving threshold (tau) updated each calibration rounds
def calibrate_mask(fisher_scores,
                   target_sparsity,
                   rounds):
    """
    Calibrate binary masks based on Fisher scores to reach target sparsity.
    The masks are iteratively refined to ensure that the target sparsity is achieved.
    Guards against keep_n out of range.

    Args:
        fisher_scores: dict mapping parameter name -> same-shape score tensor.
        target_sparsity: fraction of parameters to prune (0.0 to 1.0).
        rounds: number of calibration iterations.

    Returns:
        masks: dict mapping parameter name -> binary mask (same device/shape).
    """
    # Initialize all masks to ones (keep everything initially)
    masks = {name: torch.ones_like(scores) for name, scores in fisher_scores.items()}
    total = sum(m.numel() for m in masks.values())

    for r in range(1, rounds + 1):
        # Compute how many parameters to keep this round
        keep_frac = (1.0 - target_sparsity) ** (r / rounds)
        keep_n = int(total * keep_frac)

        # Collect scores of parameters still marked as '1' in mask
        available_scores = []
        for name in masks:
            mask_bool = masks[name].bool()
            if mask_bool.any():
                available_scores.append(fisher_scores[name][mask_bool].flatten())
        if available_scores:
            all_scores = torch.cat(available_scores)
        else:
            # No parameters left to keep
            for name in masks:
                masks[name] = torch.zeros_like(masks[name])
            break

        num_avail = all_scores.numel()

        # Guard against out-of-range keep_n
        if keep_n <= 0:
            # Prune everything
            for name in masks:
                masks[name] = torch.zeros_like(masks[name])
            break
        if keep_n >= num_avail:
            # Keep everything this round (no change)
            continue

        # Find threshold tau: the smallest score among the top-keep_n values
        tau = torch.topk(all_scores, keep_n, largest=True).values.min()

        # Rebuild mask: keep only scores <= tau (prune high-score params)
        for name in masks:
            masks[name] = (fisher_scores[name] <= tau).float()

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
