import torch


def compute_fisher_scores(model,
                          dataloader,
                          criterion,
                          device: torch.device) -> dict:
    """
    Compute Fisher Information matrix diagonal elements (sensitivity scores).
    """
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

    num_batches = len(dataloader)
    if num_batches > 0:
        for name in fisher_scores:
            fisher_scores[name] /= float(num_batches)

    return fisher_scores


# iterative pruning, moving threshold (tau) updated each calibration rounds
def calibrate_mask(fisher_scores: dict,
                   target_sparsity: float,
                   rounds: int) -> dict:
    """
    Iterative pruning: at each of `rounds` iterations, prune `target_sparsity`
    fraction of the *remaining* parameters (by Fisher score).
    """
    masks = {name: torch.ones_like(scores) for name, scores in fisher_scores.items()}

    for _ in range(rounds):
        available_scores = []
        for name, scores in fisher_scores.items():
            mask_bool = masks[name].bool()
            if mask_bool.any():
                available_scores.append(scores[mask_bool].flatten())
        if not available_scores:
            break

        all_scores = torch.cat(available_scores)
        num_avail = all_scores.numel()
        keep_n = int((1.0 - target_sparsity) * num_avail)

        if keep_n <= 0:
            for name in masks:
                masks[name] = torch.zeros_like(masks[name])
            break
        if keep_n >= num_avail:
            continue

        tau = torch.topk(all_scores, keep_n, largest=True).values.min()
        for name, scores in fisher_scores.items():
            current_mask = masks[name]
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
