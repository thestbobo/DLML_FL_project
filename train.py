import os
import yaml
import wandb
import random

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from models.dino_ViT_b16 import DINO_ViT
from data.prepare_data import get_cifar100_loaders
from project_utils.metrics import get_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, curr_epoch, verbose=False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_outputs, all_labels = [], []

    loader = tqdm(dataloader, desc=f"Train Epoch {curr_epoch}", leave=False) if verbose else dataloader

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):  # <<< AMP-enabled forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # outputs = model(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += torch.sum(predicted.eq(labels)).item()

        all_outputs.append(outputs)
        all_labels.append(labels)

    avg_loss = running_loss / total
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = get_metrics(all_outputs, all_labels)  # Calcola le metriche

    return avg_loss, metrics


def validate(model, dataloader, criterion, device, verbose=False):
    model.eval()
    running_loss, total = 0.0, 0
    all_outputs, all_labels = [], []

    loader = tqdm(dataloader, desc="Validating", leave=False) if verbose else dataloader

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

            all_outputs.append(outputs)
            all_labels.append(labels)

    avg_loss = running_loss / total
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = get_metrics(all_outputs, all_labels)  # Calcola le metriche

    return avg_loss, metrics


def main():

    # cuda status
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # load YAML config
    with open("config/config.yaml") as f:
        default_config = yaml.safe_load(f)

    # WANDB logs setup
    wandb.init(project="CIFAR-100_centralized", config=default_config)
    config = wandb.config

    set_seed(config.seed)

    # DATA
    train_loader, val_loader, test_loader = get_cifar100_loaders(config.val_split, config.batch_size,
                                                                 config.num_workers)

    # model definition
    model = DINO_ViT().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.SGD(model.classifier.parameters(),
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay,
                                momentum=config.momentum)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs - 5)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])

    # CHECKPOINT loading:
    starting_epoch = 0
    best_val_accuracy = 0.0

    if config.get("checkpoint_path", ""):
        if os.path.exists(config.checkpoint_path):
            print(f"Loading checkpoint from {config.checkpoint_path} ...")
            checkpoint = torch.load(config.checkpoint_path, map_location=device)
            model.load_state_dict((checkpoint['model_state_dict']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            starting_epoch = checkpoint['epoch']
            best_val_accuracy = checkpoint.get('val_metrics', {}).get('top1_accuracy', 0.0)
            print(f"Resumed from epoch {starting_epoch} with best val accuracy: {best_val_accuracy * 100:.2f}%")
        else:
            print(f"WARNING: Checkpoint {config.checkpoint_path} not found. Starting from scratch...")

    # TRAINING LOOP
    for epoch in range(starting_epoch, config.epochs):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch + 1,
                                                    verbose=True)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, verbose=True)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {train_loss:.4f} | Train Metrics: {train_metrics} | "
            f"Val Loss: {val_loss:.4f} | Val Metrics: {val_metrics}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        # CHECKPOINTS SAVING (and best model)
        if (epoch + 1) % 5 == 0 and config.checkpoint_save:
            checkpoint = {'epoch': epoch+1,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler_state_dict': scheduler.state_dict(),
                          'val_metrics': val_metrics,
                          'train_metrics': train_metrics}
            os.makedirs(config['out_checkpoint_dir'], exist_ok=True)
            torch.save(checkpoint, os.path.join(config['out_checkpoint_dir'], f"centralized_checkpoint_epoch_{epoch + 1}.pth"))
            print(f'Checkpoint saved at epoch {epoch + 1} with Val Metrics={val_metrics}')

        if val_metrics["top_1_accuracy"] > best_val_accuracy and config.checkpoint_save:
            best_val_accuracy = val_metrics["top_1_accuracy"]
            best_checkpoint = {'epoch': epoch + 1,
                               'model_state_dict': model.state_dict(),
                               'best_val_metrics': val_metrics,
                               'best_train_metrics': train_metrics}
            os.makedirs(config['out_checkpoint_dir'], exist_ok=True)
            torch.save(best_checkpoint, os.path.join(config['out_checkpoint_dir'], f"best_centralized_checkpoint_epoch_{epoch + 1}.pth"))
            print(f'Best model saved with Val Top-1 Accuracy={best_val_accuracy*100:.2f}%')

    print('----TRAINING COMPLETED----')


if __name__ == "__main__":
    main()
