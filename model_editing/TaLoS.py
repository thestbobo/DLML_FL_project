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
    """
    masks = {n: torch.ones_like(s) for n, s in fisher_scores.items()}
    total = sum(m.numel() for m in masks.values())
    for r in range(1, rounds+1):
        # compute round-r sparsity
        keep_frac = (1 - target_sparsity) ** (r/rounds)
        keep_n = int(total * keep_frac)

        # extract tau via top-keep_n
        all_scores = torch.cat([
            fisher_scores[n][masks[n].bool()].flatten() for n in masks
        ])
        tau = torch.topk(all_scores, keep_n, largest=True).values.min()

        # rebuild mask at this round
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
