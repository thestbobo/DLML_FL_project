import itertools
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from model_editing.SparseSGDM import SparseSGDM
from model_editing.TaLoS import compute_fisher_scores, calibrate_mask_layerwise_qk


def local_train(
    model: nn.Module,
    dataloader,
    local_steps: int,
    lr: float,
    device: torch.device,
    warmup_steps: int = 20,
):
    """
    Dense local training for exactly 'local_steps' optimizer updates.
    Args:
        model:         The PyTorch model to train.
        dataloader:    DataLoader over this client's dataset.
        local_steps:   Number of optimizer.step() calls to perform.
        lr:            Initial learning rate.
        device:        torch.device.
        warmup_steps:  Number of steps to linearly warm up (optional).
    Returns:
        (state_dict, avg_loss, accuracy)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SparseSGDM(model.parameters(), lr=lr, momentum=0.9)

    # Step‐based warmup scheduler (linear ramp for `warmup_steps` steps)
    if warmup_steps > 0:
        def lr_lambda(step_idx):
            return min((step_idx + 1) / float(warmup_steps), 1.0)
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.train()

    # Create an infinite cycling iterator over the DataLoader (for steps)
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

        # Accumulate metrics
        total_loss += loss.item() * labels.size(0)
        total_correct += outputs.argmax(1).eq(labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return model.state_dict(), avg_loss, accuracy


def local_train_talos(
    model: nn.Module,
    dataloader,
    local_steps: int,
    lr: float,
    device: torch.device,
    target_sparsity: float,
    prune_rounds: int,
    masks_dir: str,
    global_masks: dict = None,
    warmup_steps: int = 20,  # optional, step‐based warmup
):
    """
    TaLoS local training for exactly 'local_steps' optimizer updates:
      • If global_masks is provided, reuse it; else compute/load it in masks_dir.
      • After each optimizer.step(), re‐apply the (float) mask.
    Args:
        model:            PyTorch model to fine‐tune.
        dataloader:       Client’s DataLoader.
        local_steps:      Number of optimizer.step() calls.
        lr:               Initial learning rate.
        device:           torch.device.
        target_sparsity:  Fraction to prune (e.g. 0.10).
        prune_rounds:     Number of TaLoS prune iterations (R).
        masks_dir:        Directory to load/store Fisher & mask.
        global_masks:     If not None, dict[name→FloatTensor mask].
        warmup_steps:     Number of steps to warm up (optional).
    Returns:
        (state_dict, avg_loss, accuracy, global_sparsity, masks)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SparseSGDM(model.parameters(), lr=lr, momentum=0.9)

    # -----DEBUG-----
    print(">>> optimizer lr =", optimizer.param_groups[0]["lr"])
    # --------------

    # Step‐based LR scheduler for warmup (if desired)
    if warmup_steps > 0:
        def lr_lambda(step_idx):
            return min((step_idx + 1) / float(warmup_steps), 1.0)
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # === Build or load TaLoS mask ===
    os.makedirs(masks_dir, exist_ok=True)
    fisher_path = os.path.join(masks_dir, "fisher_global.pt")
    mask_path = os.path.join(masks_dir, "mask_global.pt")

    if global_masks is not None:
        masks = global_masks
    else:
        # 1) Compute or load Fisher scores (on this client or a separate loader)
        if os.path.exists(fisher_path):
            fisher_scores = torch.load(fisher_path, map_location=device)
        else:
            fisher_scores = compute_fisher_scores(model, dataloader, criterion, device)
            torch.save(fisher_scores, fisher_path)

        # 2) Compute or load mask
        if os.path.exists(mask_path):
            masks = torch.load(mask_path, map_location=device)
        else:
            keep_ratio = 1.0 - target_sparsity
            masks = calibrate_mask_layerwise_qk(
                model,
                fisher_scores,
                keep_ratio_per_block=keep_ratio,
                max_rounds=prune_rounds
            )
            torch.save(masks, mask_path)

    # 3) Apply mask once before any training begins
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(param.device))

    # DEBUG: Inspect one block’s weights to ensure they are not all zero
    with torch.no_grad():
        sample_name = f"model.blocks.0.attn.qkv.weight"
        w = model.state_dict()[sample_name]
        nonzero = (w.abs() > 1e-9).sum().item()
        total = w.numel()
        print(f">>> [DEBUG] After initial mask apply → {sample_name}: {nonzero}/{total} nonzero")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.train()

    # Infinite DataLoader for steps
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

        # Re‐apply mask immediately
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks:
                    param.mul_(masks[name].to(param.device))

        # Accumulate metrics
        total_loss += loss.item() * labels.size(0)
        total_correct += outputs.argmax(1).eq(labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Compute global sparsity
    total_params = 0
    kept_params = 0
    for mask in masks.values():
        total_params += mask.numel()
        kept_params += int(mask.sum().item())
    global_sparsity = 1.0 - (kept_params / total_params) if total_params > 0 else 0.0

    return model.state_dict(), avg_loss, accuracy, global_sparsity, masks
