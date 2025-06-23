# 📁 client.py – Clustered Conflict-Aware In-Place Masking (no TaLoS)

import itertools
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def local_train_with_mask(
    model: nn.Module,
    dataloader,
    local_steps: int,
    lr: float,
    device: torch.device,
    global_masks: dict,
    warmup_steps: int = 20,
):
    """
    Standard local training with in-place masking based on a precomputed cluster-level mask.
    No pruning rounds, no TaLoS. Just apply the mask and keep it enforced.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if warmup_steps > 0:
        def lr_lambda(step_idx):
            return min((step_idx + 1) / float(warmup_steps), 1.0)
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    if global_masks is None:
        raise ValueError("Missing global_masks for cluster-based editing.")

    masks = global_masks

    # Apply mask once before training
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(param.device))

    total_loss, total_correct, total_samples = 0.0, 0, 0
    model.train()
    infinite_loader = itertools.cycle(dataloader)

    for step in range(local_steps):
        inputs, labels = next(infinite_loader)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Re-apply mask after each step
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks:
                    param.mul_(masks[name].to(param.device))

        total_loss += loss.item() * labels.size(0)
        total_correct += outputs.argmax(1).eq(labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return model.state_dict(), avg_loss, accuracy
