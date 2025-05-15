import os
import wandb
import yaml
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from model_editing.TaLoS import compute_fisher_scores, calibrate_mask
from model_editing.SparseSGDM import SparseSGDM
from model_editing.LoRA import LoraConfig, apply_lora, get_lora_params

from models.dino_ViT_b16 import DINO_ViT
from data.prepare_data import get_cifar100_loaders, split_mask_calibration, get_sparse_loaders
from project_utils.metrics import get_metrics




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

    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # LOAD CONFIG
    with open("config/config.yaml") as f:
        default_config = yaml.safe_load(f)

    # WandB SETUP
    wandb.init(project="CIFAR-100_centralized", config=default_config)
    config = wandb.config

    # DATA LOADERS
    train_loader, val_loader, test_loader = get_cifar100_loaders(config.val_split, config.batch_size,
                                                                 config.num_workers)

    # MODEL DEFINITION
    model = DINO_ViT().to(device)

    # CRITERION, OPTIMIZER, SCHEDULER DEFINITION
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()

    # optimizer = torch.optim.SGD(model.classifier.parameters(),
    #                             lr=config.learning_rate,
    #                             weight_decay=config.weight_decay,
    #                             momentum=config.momentum)

    # LR SCHEDULER
    # warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs - 5)
    # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])

    # FINE-TUNING SETUP
    if config.finetuning_method == "lora":
        # Configure LoRA
        lora_cfg = LoraConfig(
            r=config.lora_rank,
            alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,  # e.g. ["q_proj","k_proj","v_proj","out_proj"]
            dropout=config.lora_dropout,
        )
        # Wrap the model
        print("[LoRA] applying LoRA layers to the model...")
        model = apply_lora(model, lora_cfg)
        model = model.to(device)
        torch.cuda.empty_cache()
        # Freeze backbone, leave only LoRA A/B trainable
        print("[LoRA] freezing backbone (only leaving LoRA layers trainable...")
        for name, p in model.named_parameters():
            p.requires_grad = False
        for p in get_lora_params(model):
            p.requires_grad = True

        # Build optimizer over just LoRA params
        optimizer = SparseSGDM(
            get_lora_params(model),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            mask=None,  # or supply a mask to sparsify adapters further
            model=None
        )

    elif config.finetuning_method == "dense":
        torch.cuda.empty_cache()
        optimizer = torch.optim.SGD(
            model.classifier.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    elif config.finetuning_method == "talos":
        # splits train loader to get a fraction of the data for mask calibration
        sparse_train_loader, calib_loader = get_sparse_loaders(
            train_loader.dataset,
            calib_frac=config.calib_split,
            batch_size=config.batch_size,
            calib_batch_size=config.calib_batch_size,
            num_workers=config.num_workers,
            seed=config.seed
        )
        train_loader = sparse_train_loader

        print("[TaLoS] Computing Fisher scores on calibration set...")
        fisher_scores = compute_fisher_scores(model, calib_loader, criterion, device)
        print(f"[TaLoS] Calibrating mask (sparsity={config.target_sparsity})...")
        name_mask = calibrate_mask(model, fisher_scores, config.target_sparsity, rounds=config.calib_rounds)
        torch.cuda.empty_cache()

        # Map parameter objects to mask tensors
        param_mask = {
            param: name_mask[name]
            for name, param in model.named_parameters()
            if name in name_mask
        }
        optimizer = SparseSGDM(
            params=model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            dampening=config.dampening,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay,
            mask=param_mask
        )
    else:
        raise ValueError(f"Unknown finetuning_method: {config.finetuning_method}")

    # Rebuild scheduler on new optimizer
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=config.epochs - 5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    # CHECKPOINT LOADING FOR MODEL WEIGHTS (optional: must be enabled in config.yaml):
    starting_epoch = 0
    best_val_accuracy = 0.0

    ckpt_path = config.get("checkpoint_path", "")
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path} …")
        ckpt = torch.load(ckpt_path, map_location=device)

        method = ckpt.get("finetuning_method", "").lower()
        strict = (method != "lora")
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        starting_epoch = ckpt.get("epoch", 0)
        best_val_accuracy = ckpt.get("val_metrics", {}).get("top_1_accuracy", 0.0)
        print(f"Resumed from epoch {starting_epoch} with best val@1 = {best_val_accuracy * 100:.2f}%")
    else:
        print("No valid checkpoint found; starting from scratch.")

    # Save base state for tau (task vector) extraction
    base_state = {n: p.clone().cpu() for n, p in model.named_parameters()}

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

        # CHECKPOINTS SAVING
        if (epoch + 1) % 5 == 0:
            checkpoint = {'epoch': epoch+1,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler_state_dict': scheduler.state_dict(),
                          'val_metrics': val_metrics,
                          'train_metrics': train_metrics,
                          'finetuning_method': config.finetuning_method}
            os.makedirs(config['out_checkpoint_dir'], exist_ok=True)
            torch.save(checkpoint, os.path.join(config['out_checkpoint_dir'], f"centralized_checkpoint_epoch_{epoch + 1}.pth"))
            print(f'Checkpoint saved at epoch {epoch + 1} with Val Metrics={val_metrics}')

        # BEST CHECKPOINT SAVING
        if val_metrics["top_1_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["top_1_accuracy"]
            best_checkpoint = {'epoch': epoch + 1,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'scheduler_state_dict': scheduler.state_dict(),
                               'best_val_metrics': val_metrics,
                               'best_train_metrics': train_metrics,
                               'finetuning_method': config.finetuning_method}
            os.makedirs(config['out_checkpoint_dir'], exist_ok=True)
            torch.save(best_checkpoint, os.path.join(config['out_checkpoint_dir'], f"best_centralized_checkpoint_epoch_{epoch + 1}.pth"))
            print(f'Best model saved with Val Top-1 Accuracy={best_val_accuracy*100:.2f}%')

    # EXTRACT SPARSE TASK VECTOR (TAU)
    if config.finetuning_method.lower() == "talos":
        print("[TaLoS] Extracting task vector (tau)...")
        tau = {}
        for name, param in model.named_parameters():
            if name in name_mask:
                mask_cpu = name_mask[name].cpu()
                diff = (param.cpu() - base_state[name])
                tau[name] = diff.mul(name_mask[name])
        torch.save(tau, Path(config.out_checkpoint_dir) / "sparse_task_vector.pt")
        # Save task vector
        os.makedirs(config['out_checkpoint_dir'], exist_ok=True)
        torch.save(tau, os.path.join(config['out_checkpoint_dir'], "sparse_task_vector.pth"))
        print(f"[TaLoS] Task Vector saved at {os.path.join(config['out_checkpoint_dir'], 'sparse_task_vector.pth')}")

    print('----TRAINING COMPLETED----')


if __name__ == "__main__":
    main()
