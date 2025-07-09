import torch
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    LambdaLR
)

def get_scheduler(optimizer, config):
    stype = config.get("scheduler_type", "").lower()
    if stype == "cosine":
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
        cosine = CosineAnnealingLR(optimizer, T_max=config['epochs'] - 5)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
    elif stype == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
    elif stype == "step":
        return StepLR(optimizer, step_size=10, gamma=0.1)
    elif stype == "lambda":
        decay_rate = config.get("lambda_decay_rate", 0.95)
        lambda_fn = lambda epoch: decay_rate ** epoch
        return LambdaLR(optimizer, lr_lambda=lambda_fn)
    elif stype == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler_type: {stype}")