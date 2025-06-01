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
    model,
    dataloader,
    epochs: int,
    lr: float,
    device: torch.device,
    target_sparsity: float,
    prune_rounds: int,
    fisher_loader=None,
    warmup_epochs: int = 5,
    masks_dir: str = "./masks",
    client_id: int = 0,
):
    """
    TaLoS sparse fine-tuning with:
      - Fisher-score caching
      - Mask calibration & saving
      - 5-epoch warmup + cosine LR
      - Local metrics & sparsity reporting
    Returns:
        state_dict, avg_loss, accuracy, global_sparsity, masks
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Warm-up + Cosine scheduler setup
    total_epochs = epochs
    w = min(warmup_epochs, total_epochs)

    def lr_fn(epoch):
        if total_epochs <= w:
            return float(epoch + 1) / float(total_epochs)
        if epoch < w:
            return float(epoch + 1) / float(w)
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - w) / float(total_epochs - w)))

    scheduler = LambdaLR(optimizer, lr_fn)

    # Ensure cache dir
    os.makedirs(masks_dir, exist_ok=True)
    fisher_path = os.path.join(masks_dir, f"fisher_client_{client_id}.pt")
    mask_path = os.path.join(masks_dir, f"mask_client_{client_id}.pt")

    # Load / compute Fisher scores
    fl = fisher_loader or dataloader
    if os.path.exists(fisher_path):
        fisher_scores = torch.load(fisher_path, map_location=device)
    else:
        fisher_scores = compute_fisher_scores(model, fl, criterion, device)
        torch.save(fisher_scores, fisher_path)

    # Load / calibrate mask
    if os.path.exists(mask_path):
        masks = torch.load(mask_path, map_location=device)
    else:
        masks = calibrate_mask(fisher_scores, target_sparsity, prune_rounds)
        torch.save(masks, mask_path)

    # Initial mask application
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(param.device))

    # Local fine-tuning with scheduler and mask enforcement
    total_loss, total_correct, total_samples = 0.0, 0, 0
    model.train()
    for epoch in range(total_epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # re-apply mask
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        param.mul_(masks[name].to(param.device))

            # accumulate metrics
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += outputs.argmax(1).eq(y).sum().item()
            total_samples += bs

        scheduler.step()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    # Sparsity calculation
    total_params = 0
    kept_params = 0
    for mask in masks.values():
        total_params += mask.numel()
        kept_params += mask.sum().item()
    global_sparsity = 1.0 - (kept_params / total_params)

    return model.state_dict(), avg_loss, accuracy, global_sparsity, masks
