import torch

def compute_fisher_scores(model, dataloader, criterion, device):
    """
    Compute Fisher information for each parameter by accumulating squared gradients over one pass of the data.
    Returns a dict mapping parameter names to Fisher scores (tensor).
    """
    model.eval()
    fisher = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters() if param.requires_grad}
    count = 0

    for batch in dataloader:
        x, y = batch[0].to(device), batch[1].to(device)
        model.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += (param.grad.detach() ** 2)
        count += x.size(0)

    # Normalize
    for name in fisher:
        fisher[name] = fisher[name] / count

    return fisher

def calibrate_mask_global(fisher_scores, target_sparsity):
    """
    Prune parameters globally by retaining the lowest Fisher scores until target sparsity is reached.
    Returns a dict mapping parameter names to binary mask tensors.
    """
    # Flatten all Fisher scores into a single vector to determine threshold
    all_scores = torch.cat([f.view(-1) for f in fisher_scores.values()])
    k = int((1 - target_sparsity) * all_scores.numel())
    if k < 1:
        threshold = torch.inf
    else:
        threshold, _ = torch.kthvalue(all_scores, k)
    # Create masks: 1 for kept, 0 for pruned
    name_mask = {}
    for name, f in fisher_scores.items():
        mask = (f <= threshold).float()
        name_mask[name] = mask
    return name_mask