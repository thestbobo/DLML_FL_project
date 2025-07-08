from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR, LinearLR, CosineAnnealingLR, ReduceLROnPlateau

def get_scheduler(optimizer, config):
    scheduler_type = config.scheduler_type.lower()

    if scheduler_type == "cosine":
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
        cosine = CosineAnnealingLR(optimizer, T_max=config.epochs - 5)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            verbose=True
        )

    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )

    elif scheduler_type == "lambda":
        decay_rate = config.get("lambda_decay_rate", 0.95)
        lambda_fn = lambda epoch: decay_rate ** epoch
        return LambdaLR(optimizer, lr_lambda=lambda_fn)

    elif scheduler_type == "none":
        return None

    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
