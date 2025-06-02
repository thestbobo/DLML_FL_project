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
                   target_sparsity: float,
                   rounds: int):
    """
    Iterative pruning: at each of `rounds` iterations, prune `target_sparsity`
    fraction of the remaining parameters (by Fisher score).

    Args:
        fisher_scores: dict mapping param name -> same-shape score tensor.
        target_sparsity: fraction of parameters to prune each round (0.0–1.0).
        rounds: number of iterative pruning rounds.

    Returns:
        masks: dict mapping param name -> binary mask (1=keep, 0=prune).
    """
    # 1) Initialize all‐ones mask
    masks = {name: torch.ones_like(scores) for name, scores in fisher_scores.items()}

    for r in range(rounds):
        # 2) Gather all unpruned scores
        available_scores = []
        for name, scores in fisher_scores.items():
            mask_bool = masks[name].bool()
            if mask_bool.any():
                available_scores.append(scores[mask_bool].flatten())
        if not available_scores:
            break  # nothing left to prune

        all_scores = torch.cat(available_scores)
        num_avail = all_scores.numel()

        # 3) Compute how many to keep this round: (1 - target_sparsity) fraction of remaining
        keep_n = int((1.0 - target_sparsity) * num_avail)

        # 4) Guard against out-of-range
        if keep_n <= 0:
            # prune everything
            for name in masks:
                masks[name] = torch.zeros_like(masks[name])
            break
        if keep_n >= num_avail:
            # keep everything (no pruning this iteration)
            continue

        # 5) Find threshold among top‐keep_n
        tau = torch.topk(all_scores, keep_n, largest=True).values.min()

        # 6) Update each mask: keep = (current_mask == 1) AND (score <= tau)
        for name, scores in fisher_scores.items():
            current_mask = masks[name]
            mask_bool = current_mask.bool()
            # Only consider scores that are currently unpruned
            new_mask = (scores <= tau).float()
            masks[name] = current_mask * new_mask

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
