from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    LambdaLR
)


def get_scheduler(optimizer, config):
    scheduler_type = config.scheduler_type.lower()
    
    if config.scheduler_type == "cosine":
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
        cosine = CosineAnnealingLR(optimizer, T_max=config.epochs - 5)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    elif config.scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            verbose=True
        )

    elif config.scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    
    elif scheduler_type == "lambda":
        decay_rate = config.get("lambda_decay_rate", 0.95)
        lambda_fn = lambda epoch: decay_rate ** epoch
        return LambdaLR(optimizer, lr_lambda=lambda_fn)
    
    elif scheduler_type == "none":
        return None  # no scheduler

    else:
        raise ValueError(f"Unknown scheduler_type: {config.scheduler_type}")
