import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from model_editing.TaLoS import calibrate_mask, compute_fisher_scores


def local_train(model, dataloader, epochs, lr, device, warmup_epochs=5):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # Warm-up + Cosine scheduler
    total_epochs = epochs
    w = min(warmup_epochs, total_epochs)

    def lr_fn(epoch):
        if total_epochs <= w:
            return float(epoch + 1) / float(total_epochs)
        if epoch < w:
            return float(epoch + 1) / float(w)
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - w) / float(total_epochs - w)))

    scheduler = LambdaLR(optimizer, lr_fn)

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for epoch in range(total_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += outputs.argmax(1).eq(labels).sum().item()
            total_samples += bs

        scheduler.step()  # step at epoch end

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return model.state_dict(), avg_loss, accuracy


# MASKS COMPUTED LOCALLY IN EACH CLIENT.
def local_train_talos(
    model: nn.Module,
    dataloader,
    epochs: int,
    lr: float,
    device: torch.device,
    target_sparsity: float,
    prune_rounds: int,
    fisher_loader=None,
    warmup_epochs: int = 5,
    masks_dir: str = "./masks",
    global_masks: dict = None,  # NEW param: if given, we skip per-client mask logic
):
    """
    TaLoS sparse fine-tuning with:
      - Fisher-score caching
      - Mask calibration & saving (once per run, not per client)
      - 5-epoch warmup + cosine LR
      - Local metrics & sparsity reporting

    Returns:
        state_dict, avg_loss, accuracy, global_sparsity, masks
    """
    # 1) Move model to device
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 2) LR scheduler (warm-up + cosine)
    total_epochs = epochs
    w = min(warmup_epochs, total_epochs)

    def lr_fn(epoch_idx: int):
        if total_epochs <= w:
            return float(epoch_idx + 1) / float(total_epochs)
        if epoch_idx < w:
            return float(epoch_idx + 1) / float(w)
        return 0.5 * (1.0 + math.cos(math.pi * (epoch_idx - w) / float(total_epochs - w)))

    scheduler = LambdaLR(optimizer, lr_fn)

    # 3) Ensure cache dir for Fisher scores
    os.makedirs(masks_dir, exist_ok=True)
    fisher_path = os.path.join(masks_dir, f"fisher_global.pt")  # single Fisher
    mask_path = os.path.join(masks_dir, f"mask_global.pt")      # single mask

    # 4) Load or compute Fisher scores (once per run)
    fl = fisher_loader or dataloader
    if os.path.exists(fisher_path):
        fisher_scores = torch.load(fisher_path, map_location=device)
    else:
        # We compute Fisher scores on the *unpruned* model once
        fisher_scores = compute_fisher_scores(model, fl, criterion, device)
        torch.save(fisher_scores, fisher_path)

    # 5) Decide which mask to use
    if global_masks is not None:
        # We were given a precomputed mask from train_federated.py
        masks = global_masks
    else:
        # Fallback: if no global_masks passed, recompute once and save
        if os.path.exists(mask_path):
            masks = torch.load(mask_path, map_location=device)
        else:
            masks = calibrate_mask(fisher_scores, target_sparsity, prune_rounds)
            torch.save(masks, mask_path)

    # 6) Apply the mask once before any training
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(param.device))

    # 7) Local fine-tuning loop (enforce mask each step)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    model.train()

    for epoch_idx in range(total_epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # Immediately re-apply mask in-place
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        param.mul_(masks[name].to(param.device))

            # Accumulate metrics
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += outputs.argmax(1).eq(y).sum().item()
            total_samples += bs

        scheduler.step()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # 8) Compute overall sparsity
    total_params = 0
    kept_params = 0
    for mask in masks.values():
        total_params += mask.numel()
        kept_params += mask.sum().item()
    global_sparsity = 1.0 - (kept_params / total_params) if total_params > 0 else 0.0

    return model.state_dict(), avg_loss, accuracy, global_sparsity, masks
