import math

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
        if epoch < w:
            # linear warm-up: (epoch+1)/w
            return float(epoch + 1) / float(w)
        else:
            # cosine annealing over [w, total_epochs)
            return 0.5 * (1.0 + math.cos(
                math.pi * (epoch - w) / float(total_epochs - w)
            ))

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
    warmup_epochs: int = 5
):
    """
    TALos‐style sparse fine-tuning for FL:
      1) Compute Fisher scores per-parameter on fisher_loader (or dataloader).
      2) Calibrate a binary mask to keep top (1 - target_sparsity) weights.
      3) Apply the mask and fine-tune, re-applying the mask after each update.
    Returns:
        The pruned-and-fine-tuned state_dict.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Warm-up + Cosine scheduler setup
    total_epochs = epochs
    w = min(warmup_epochs, total_epochs)

    def lr_fn(epoch):
        if epoch < w:
            return float(epoch + 1) / float(w)
        else:
            return 0.5 * (1.0 + math.cos(
                math.pi * (epoch - w) / float(total_epochs - w)
            ))

    scheduler = LambdaLR(optimizer, lr_fn)

    # Compute Fisher scores
    fl = fisher_loader or dataloader
    fisher_scores = compute_fisher_scores(model, fl, criterion, device)

    # Build and apply binary mask
    masks = calibrate_mask(fisher_scores, target_sparsity, prune_rounds)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name])

    # Local fine-tuning with scheduler and mask enforcement
    model.train()
    for epoch in range(total_epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            # re-apply mask to enforce sparsity
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        param.mul_(masks[name])

        scheduler.step()

    return model.state_dict()
