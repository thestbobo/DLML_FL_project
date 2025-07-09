import torch

def compute_fisher_mask(model, dataloader, device, threshold):
    """
    Compute a Fisher-information-based mask for pruning.
    Args:
        model: nn.Module
        dataloader: data loader for samples to use
        device: torch.device
        threshold: float, threshold for pruning
    Returns:
        mask: dict mapping nn.Parameter to binary mask tensor
    """
    fisher = {p: torch.zeros_like(p, device=device) for p in model.parameters() if p.requires_grad}
    model.eval()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                fisher[p] += (p.grad.detach() ** 2)
    # normalize by dataset size
    for p in fisher:
        fisher[p] = fisher[p] / len(dataloader.dataset)
    # build mask: keep weights with Fisher below threshold (for demo, usually it's the other way)
    mask = {}
    for p, f in fisher.items():
        mask[p] = (f < threshold).float()
    return mask