import torch
import torch.nn as nn
import torch.optim as optim

from model_editing.TaLoS import calibrate_mask, compute_fisher_scores


def local_train(model, dataloader, epochs, lr, device):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate metrics
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += batch_size
        scheduler.step()

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
    fisher_loader=None
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

    # Compute Fisher scores
    fl = fisher_loader or dataloader
    fisher_scores = compute_fisher_scores(model, fl, criterion, device)

    # Build binary mask
    masks = calibrate_mask(fisher_scores, target_sparsity, prune_rounds)

    # Apply initial mask
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name])

    # Local fine-tuning with mask applied
    model.train()
    for _ in range(epochs):
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

    return model.state_dict()
