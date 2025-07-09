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

def calibrate_mask_global(fisher_scores, target_sparsity, whitelist=None, min_keep_frac=0.5):
    """
    Args:
        fisher_scores: dict param_name -> tensor of Fisher scores
        target_sparsity: global sparsity
        whitelist: list of substrings. Any param whose name contains one is whitelisted.
        min_keep_frac: minimum fraction to keep in whitelisted layers
    """
    all_scores = torch.cat([f.view(-1) for f in fisher_scores.values()])
    k = int((1 - target_sparsity) * all_scores.numel())
    threshold = torch.inf if k < 1 else torch.kthvalue(all_scores, k).values.item()

    name_mask = {}
    for name, f in fisher_scores.items():
        is_whitelisted = any(w in name for w in (whitelist or []))
        if is_whitelisted:
            flat = f.view(-1)
            num_to_keep = int(min_keep_frac * flat.numel())
            if num_to_keep < 1:
                num_to_keep = 1
            if num_to_keep >= flat.numel():
                mask = torch.ones_like(flat)
            else:
                local_threshold = torch.kthvalue(flat, num_to_keep).values.item()
                mask = (flat <= local_threshold).float()
            mask = mask.view_as(f)
        else:
            mask = (f <= threshold).float()
        name_mask[name] = mask
    return name_mask

