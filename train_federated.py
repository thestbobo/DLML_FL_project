import os
import copy
import time
import numpy as np
import torch
import yaml
import wandb
from torch.utils.data import DataLoader, ConcatDataset
import re

from models.dino_ViT_b16 import DINO_ViT
from data import get_client_datasets, get_test_loader
from model_editing.mask_manager import MaskManager
from model_editing.TaLoS import compute_fisher_scores, calibrate_mask_global
from fl_core.client import Client
from fl_core.server import Server
from project_utils.metrics import get_metrics
from project_utils.logger import PrintLogger, CheckpointManager
from project_utils.wandb_logger import WandBLogger

def evaluate(model, dataloader, device):
    model.eval().to(device)
    all_outputs, all_labels = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            all_outputs.append(outputs)
            all_labels.append(y)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = get_metrics(all_outputs, all_labels)
    avg_loss = total_loss / len(dataloader.dataset)
    metrics["global_loss"] = avg_loss
    return metrics

def save_checkpoint(model, round_num, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"fl_model_round_{round_num}.pth")
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved checkpoint to {path}")

def load_checkpoint(model, path):
    if not os.path.exists(path):
        print("[WARN] No checkpoint found.")
        return 0
    print(f"[INFO] Loading checkpoint from {path} …")
    ckpt = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        round_num = ckpt.get("round", 0)
    else:
        model.load_state_dict(ckpt, strict=False)
        match = re.search(r"round_(\d+)", path)
        round_num = int(match.group(1)) if match else 0
    print(f"[INFO] Resumed from round {round_num}")
    return round_num

def compute_dynamic_sparsity(init_sparsity, final_sparsity, current_round, total_rounds):
    return init_sparsity + (final_sparsity - init_sparsity) * (current_round / total_rounds)

def print_mask_stats(mask_dict):
    total_params = total_kept = 0
    print("[MASK STATS]")
    for name, mask in mask_dict.items():
        kept = int(mask.sum())
        total = mask.numel()
        pct = 100 * kept / total
        print(f"{name:45s}: kept {kept:7d}/{total:7d} ({pct:5.1f}%)")
        total_params += total
        total_kept += kept
    print(f"Total kept: {total_kept}/{total_params} ({100*total_kept/total_params:.2f}%)")


def main():
    # Load config and initialize loggers
    with open("config/config.yaml", encoding="utf-8") as f:
        default_config = yaml.safe_load(f)

    wandb_logger = WandBLogger(
        project="Federated-DINO-ViT",
        run_id=default_config.get("run_id", None),
        config=default_config
    )
    logger = PrintLogger()
    config = wandb.config

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    method = config.FINETUNE_METHOD.lower()
    masks_root = config.get("MASKS_DIR", "masks/")
    mask_manager = None
    shared_masks = None

    # Model initialization
    if method == "talos":
        global_model = DINO_ViT(num_classes=100, pretrained=True, frozen_backbone=False)
    else:
        global_model = DINO_ViT(num_classes=100, pretrained=True, frozen_backbone=True)

    # Resume from checkpoint if available
    ckpt_path = config.CHECKPOINT_PATH
    starting_round = load_checkpoint(global_model, ckpt_path)

    # Data preparation
    client_datasets = get_client_datasets(config.IID, config.NUM_CLIENTS, config.NC, config.seed)
    
    from collections import Counter
    for i, ds in enumerate(client_datasets):
        labels = [y for _, y in ds]
        print(f"Client {i}: {len(ds)} samples, label counts: {Counter(labels)}")
    
    test_loader = get_test_loader(batch_size=config.BATCH_SIZE)

    # Mask preparation (for TaLoS)
    if method == "talos":
        mask_manager = MaskManager(config, global_model, device)
        mask_file = os.path.join(masks_root, "mask_global.pt")
        need_to_compute_mask = not os.path.exists(mask_file)
        if need_to_compute_mask:
            print("\n>>> Building calibration loader over full CIFAR-100 training set …")
            full_train_dataset = ConcatDataset(client_datasets)
            fisher_loader = DataLoader(
                full_train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True
            )
            print("\n>>> Computing Fisher scores …")
            dummy_model = copy.deepcopy(global_model).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            fisher_scores = compute_fisher_scores(dummy_model, fisher_loader, criterion, device)
            print(">>> Calibrating global mask (least sensitive params kept) …")
            shared_masks = calibrate_mask_global(
                fisher_scores,
                target_sparsity=config.TALOS_TARGET_SPARSITY, 
                whitelist=["classifier", "pos_embed", "patch_embed", "blocks.0.", "blocks.11."],
                min_keep_frac=0.5
            )
            mask_manager.save_mask(shared_masks, "mask_global.pt")
            del dummy_model, fisher_scores
        else:
            shared_masks = mask_manager.load_mask("mask_global.pt")
        print(f">>> Shared mask ready at {masks_root}")
        
   
    # After mask creation
    print_mask_stats(shared_masks)
    
    total, kept = 0, 0
    for name, mask in shared_masks.items():
        print(f"{name:40s}: kept {int(mask.sum())}/{mask.numel()} ({100*mask.sum()/mask.numel():.2f}%)")
        total += mask.numel()
        kept += int(mask.sum())
    print(f"Total kept: {kept}/{total} ({100*kept/total:.2f}%)")
    
    # Federated training loop
    server = Server()
    checkpoint_manager = CheckpointManager(config.OUT_CHECKPOINT_PATH)
    for t_round in range(starting_round + 1, config.ROUNDS + 1):
        print(f"\n--- Round {t_round} ---")
        m = max(int(config.CLIENT_FRACTION * config.NUM_CLIENTS), 1)
        selected_clients = np.random.choice(config.NUM_CLIENTS, m, replace=False)
        if t_round % 5 == 0:
            wandb_logger.log_aggregated_class_distribution(client_datasets, selected_clients, t_round)

        prev_global_weights = copy.deepcopy(global_model.state_dict())
        local_weights, num_samples_list = [], []
        client_to_log = np.random.choice(selected_clients)
        lr_round = config.LR * (config.LR_DECAY ** (t_round - 1))

        for cid in selected_clients:
            print(f"Training client -> {cid}")
            dataset = client_datasets[cid]
            loader = DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            # Construct client and perform local update
            client = Client(
                cid=cid,
                model=copy.deepcopy(global_model),
                train_loader=loader,
                config={
                    "FEDERATED_TRAINING": {
                        "LR": lr_round,
                        "LOCAL_STEPS": config.LOCAL_STEPS,
                        "LR_DECAY": config.LR_DECAY
                    },
                    "talos_fine_tuning": {"nesterov": 0.9} if method == "talos" else {},
                },
                device=device,
                mask=shared_masks if method == "talos" else None
            )
            local_state_dict = client.local_update(global_model.state_dict())
            local_weights.append(local_state_dict)
            num_samples_list.append(len(dataset))

            # Optionally: log client metrics (dummy, extend as needed)
            if cid == client_to_log:
                logger.log_client_metrics(cid, {"dummy_metric": 0.0}, t_round)

        # Aggregate with (weighted) FedAvg
        global_weights = server.aggregate(local_weights, num_samples_list)
        global_model.load_state_dict(global_weights)
        wandb_logger.log_global_weight_diff(prev_global_weights, global_weights, t_round)

        # Evaluate global model
        metrics = evaluate(global_model, test_loader, device)
        wandb_logger.log_global_metrics(metrics, t_round)
        logger.log_global_metrics(metrics, t_round)
        wandb_logger.log_round_info(
            t_round,
            len(selected_clients),
            sum(len(client_datasets[cid]) for cid in selected_clients)
        )

        if t_round % 5 == 0 or t_round == config.ROUNDS:
            checkpoint_manager.save_checkpoint(global_model, t_round)

        print(f"Round {t_round} complete — Global loss: {metrics['global_loss']:.4f}, metrics: {metrics}\n")

    print(f"\n{'=' * 10} Training Completed {'=' * 10}")
    return global_model

if __name__ == "__main__":
    main()