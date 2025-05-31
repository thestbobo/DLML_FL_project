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
def calibrate_mask(fisher_scores, target_sparsity, rounds):
    """
    Calibrate binary masks based on Fisher scores to reach target sparsity.
    The masks are iteratively refined to ensure that the target sparsity is achieved.
    Guards against keep_n out of range.
    """
    # Initialize masks to all ones
    masks = {name: torch.ones_like(scores) for name, scores in fisher_scores.items()}
    total = sum(m.numel() for m in masks.values())

    for r in range(1, rounds + 1):
        # compute round-r sparsity
        keep_frac = (1 - target_sparsity) ** (r / rounds)
        keep_n = int(total * keep_frac)

        # gather scores currently unmasked
        available_scores = []
        for n in masks:
            # only include scores where mask is 1
            available_scores.append(fisher_scores[n][masks[n].bool()].flatten())
        if available_scores:
            all_scores = torch.cat(available_scores)
        else:
            # no parameters left to keep
            for n in masks:
                masks[n] = torch.zeros_like(masks[n])
            break

        num_avail = all_scores.numel()
        # guard against keep_n out of range
        if keep_n <= 0:
            for n in masks:
                masks[n] = torch.zeros_like(masks[n])
            break
        if keep_n >= num_avail:
            # nothing to prune this round, skip to next iteration
            continue

        # compute threshold tau from top-keep_n scores
        tau = torch.topk(all_scores, keep_n, largest=True).values.min()

        # rebuild mask at this round: keep scores <= tau (prune high scores first)
        for n in masks:
            masks[n] = (fisher_scores[n] <= tau).float()

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
