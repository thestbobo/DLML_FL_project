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
    masks_dir: str,
    global_masks: dict = None,
    warmup_epochs: int = 5,
):
    """
    If `global_masks` is provided, reuse it.
    Otherwise, load or compute a single Fisher mask in `masks_dir`.
    """

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    total_epochs = epochs
    w = min(warmup_epochs, total_epochs)

    def lr_fn(epoch_idx: int):
        if total_epochs <= w:
            return (epoch_idx + 1) / float(total_epochs)
        if epoch_idx < w:
            return (epoch_idx + 1) / float(w)
        return 0.5 * (1.0 + math.cos(math.pi * (epoch_idx - w) / float(total_epochs - w)))

    scheduler = LambdaLR(optimizer, lr_fn)

    os.makedirs(masks_dir, exist_ok=True)
    fisher_path = os.path.join(masks_dir, "fisher_global.pt")
    mask_path   = os.path.join(masks_dir, "mask_global.pt")

    # ─── Decide mask source ───
    if global_masks is not None:
        masks = global_masks

    else:
        # 1) Load or compute Fisher scores
        if os.path.exists(fisher_path):
            fisher_scores = torch.load(fisher_path, map_location=device)
        else:
            fisher_scores = compute_fisher_scores(model, dataloader, criterion, device)
            torch.save(fisher_scores, fisher_path)

        # 2) Load or compute mask
        if os.path.exists(mask_path):
            masks = torch.load(mask_path, map_location=device)
        else:
            masks = calibrate_mask(fisher_scores, target_sparsity, prune_rounds)
            torch.save(masks, mask_path)

    # ─── Apply the mask once before training ───
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(param.device))

    # ─── Local training loop ───
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.train()

    for epoch_idx in range(total_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Re‐apply mask in‐place every step
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        param.mul_(masks[name].to(param.device))

            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct = (outputs.argmax(dim=1) == labels).sum().item()
            total_correct += correct
            total_samples += bs

        scheduler.step()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Calculate global sparsity = fraction of weights pruned
    total_params = 0
    kept_params = 0
    for mask in masks.values():
        total_params += mask.numel()
        kept_params += int(mask.sum().item())
    global_sparsity = 1.0 - (kept_params / total_params) if total_params > 0 else 0.0

    return model.state_dict(), avg_loss, accuracy, global_sparsity, masks
