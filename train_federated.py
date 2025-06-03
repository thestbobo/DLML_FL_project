import os
import copy
import numpy as np
import torch
import yaml
import wandb
from torch.utils.data import DataLoader, ConcatDataset, Subset
import re
from model_editing.TaLoS import compute_fisher_scores, calibrate_mask, calibrate_mask_layerwise_qk
from models.dino_ViT_b16 import DINO_ViT
from fl_core.client import local_train, local_train_talos
from fl_core.server import average_weights_fedavg
from data.prepare_data_fl import get_client_datasets, get_test_loader
from project_utils.metrics import get_metrics
from project_utils.federated_metrics import (
    log_global_weight_diff,
    log_aggregated_class_distribution,
    log_round_info,
    log_global_metrics,
    log_client_metrics,
)


def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def main():
    # load config / init WandB
    with open("config/config.yaml") as f:
        default_config = yaml.safe_load(f)

    wandb.init(project="Federated-DINO-ViT", config=default_config)
    config = wandb.config

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    method = config.FINETUNE_METHOD.lower()

    # TaLoS only -> manage mask cache paths
    if method == "talos":
        if config.LOAD_MASK:
            # User specified an existing mask folder (or file) to load
            masks_root = config.LOAD_MASK
            os.makedirs(masks_root, exist_ok=True)
            print(f">>> Loading precomputed mask from: {masks_root}")
            need_to_compute_mask = False
        else:
            # No pre‐computed mask: we will compute it from scratch & save under ./masks_run
            masks_root = config.MASKS_DIR
            os.makedirs(masks_root, exist_ok=True)
            print(f">>> No LOAD_MASK set; computing new mask and storing under: {masks_root}")
            need_to_compute_mask = True

        global_fisher_file = os.path.join(masks_root, "fisher_global.pt")
        global_mask_file = os.path.join(masks_root, "mask_global.pt")
    else:
        masks_root = None
        need_to_compute_mask = False
        global_fisher_file = None
        global_mask_file = None

    mode = "IID" if config.IID else f"Non-IID Nc={config.NC}"
    print(f"========== Federated Training Start ({mode}) ==========")

    starting_round = 0
    best_test_accuracy = 0.0
    ckpt_path = config.CHECKPOINT_PATH

    # building model / loading existing one from config
    global_model = DINO_ViT(num_classes=100, pretrained=True)
    print("[INFO] Loading from checkpoint:", ckpt_path)
    print("[INFO] Exists?", os.path.exists(ckpt_path))

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[INFO] Loading checkpoint from {ckpt_path} …")
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            method_ckpt = ckpt.get("finetuning_method", "").lower()
            strict = (method_ckpt != "lora")
            global_model.load_state_dict(ckpt["model_state_dict"], strict=strict)
            starting_round = ckpt.get("round", 0)
            best_test_accuracy = ckpt.get("test_metrics", {}).get("top_1_accuracy", 0.0)
            print(f"[INFO] Resumed from round {starting_round} with best top-1 acc = {best_test_accuracy:.2%}")
        else:
            print("[WARN] Checkpoint is a pure state_dict. Loading weights only.")
            global_model.load_state_dict(ckpt, strict=False)
            match = re.search(r"round_(\d+)", ckpt_path)
            starting_round = int(match.group(1)) if match else 0
            print(f"[INFO] Resumed with state_dict only — starting from round {starting_round}")
    else:
        print("No valid checkpoint found; starting from scratch.")

    # data prep
    client_datasets = get_client_datasets(config.IID, config.NUM_CLIENTS, config.seed)
    test_loader = get_test_loader(batch_size=config.BATCH_SIZE)

    # talos branch
    if method == "talos":
        if need_to_compute_mask:
            print("\n>>> Building a calibration loader over the FULL CIFAR-100 training set …")
            # Instead of sampling a few clients, we simply concatenate all client splits.
            # This recreates the entire CIFAR-100 training set (assuming get_client_datasets splits it disjointly).
            full_train_dataset = ConcatDataset(client_datasets)
            calib_loader = DataLoader(
                full_train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
        else:
            calib_loader = None

        print("\n>>> Preparing shared Fisher + mask (TaLoS) …")
        if need_to_compute_mask:
            # Compute Fisher scores on the entire dataset
            dummy = copy.deepcopy(global_model).to(device)
            dummy_criterion = torch.nn.CrossEntropyLoss()

            fisher_scores = compute_fisher_scores(dummy, calib_loader, dummy_criterion, device)
            torch.save(fisher_scores, global_fisher_file)

            # Calibrate the binary mask over multiple rounds

            # ----- other way of computing mask -----
            # R = config.TALOS_PRUNE_ROUNDS
            # shared_masks = calibrate_mask(
            #     fisher_scores,
            #     target_sparsity=config.TALOS_TARGET_SPARSITY,
            #     rounds=R
            # )
            # ---------------------------------------

            # Build a layer‐wise Q/K float mask (10 % keep)
            shared_masks = calibrate_mask_layerwise_qk(
                dummy,
                fisher_scores,
                keep_ratio_per_block=0.10
            )

            torch.save(shared_masks, global_mask_file)
            del dummy, fisher_scores

        else:
            # Just load what’s already on disk
            fisher_scores = torch.load(global_fisher_file, map_location=device)
            shared_masks = torch.load(global_mask_file, map_location=device)

        print(
            f">>> Shared mask ready ({'loaded from' if not need_to_compute_mask else 'computed and saved to'}) → {masks_root}")

    else:
        shared_masks = None
        print("\n>>> Skipping Fisher/mask preparation — dense training.")

    # lr.
    base_lr = config.LR
    decay = config.LR_DECAY
    warmup_eps = config.WARMUP_EPOCHS

    # federated loop
    for t_round in range(starting_round + 1, config.ROUNDS + 1):
        print(f"\n--- Round {t_round} ---")

        # 7.1 Select a subset of clients
        m = max(int(config.CLIENT_FRACTION * config.NUM_CLIENTS), 1)
        selected_clients = np.random.choice(config.NUM_CLIENTS, m, replace=False)

        # 7.2 Log aggregated class distribution every 5 rounds
        if t_round % 5 == 0:
            log_aggregated_class_distribution(client_datasets, selected_clients, t_round)

        prev_global_weights = global_model.state_dict()
        local_weights, num_samples_list = [], []
        client_to_log = np.random.choice(selected_clients)  # pick one client for local‐metrics logging
        lr_round = base_lr * (decay ** (t_round - 1))

        for cid in selected_clients:
            print(f"Training client -> {cid}")

            # Show dataset size for debugging
            cnt = sum(1 for _ in DataLoader(client_datasets[cid], batch_size=1))
            print(f"  Client {cid} dataset size: {cnt}")

            local_model = copy.deepcopy(global_model)
            loader = DataLoader(
                client_datasets[cid],
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )

            # Keep track of initial weights to log weight‐delta (L2)
            initial_weights = copy.deepcopy(local_model.state_dict())
            method = config.FINETUNE_METHOD.lower()

            if method == "dense":
                w, avg_loss, acc = local_train(
                    local_model,
                    loader,
                    epochs=config.LOCAL_EPOCHS,
                    lr=lr_round,
                    device=device,
                    warmup_epochs=warmup_eps
                )
                sparsity = None
                masks = None

            elif method == "talos":
                # ─── Always use the SINGLE precomputed global mask ───
                w, avg_loss, acc, sparsity, masks = local_train_talos(
                    local_model,
                    loader,
                    epochs=config.LOCAL_EPOCHS,
                    lr=lr_round,
                    device=device,
                    target_sparsity=config.TALOS_TARGET_SPARSITY,
                    prune_rounds=config.TALOS_PRUNE_ROUNDS,
                    masks_dir=masks_root,  # pass the same root where we saved global_mask
                    global_masks=shared_masks,  # force‐use the precomputed global mask
                    warmup_epochs=warmup_eps
                )
            else:
                raise ValueError(f"Unknown FINETUNE_METHOD '{method}'")

            # Compute the L2‐norm of weight update (for logging)
            diff_norm = 0.0
            for name in initial_weights:
                p0 = initial_weights[name].cpu()
                p1 = w[name].cpu()
                diff_norm += (p0 - p1).norm().item() ** 2
            diff_norm = diff_norm ** 0.5
            print(f"  Client {cid} local weight diff L2: {diff_norm:.6f}")
            print(f"  Client {cid} local loss={avg_loss:.4f}, acc={acc:.4f}")
            if sparsity is not None:
                print(f"  Client {cid} sparsity={100 * sparsity:.2f}%")

            local_weights.append(w)
            num_samples_list.append(len(client_datasets[cid]))

            # Log this client’s metrics when it was randomly selected
            if cid == client_to_log:
                log_client_metrics(cid, avg_loss, acc, t_round)
                if sparsity is not None:
                    # Log global sparsity and layer‐wise density histogram to W&B
                    wandb.log({
                        "round": t_round,
                        f"client_{cid}/sparsity": sparsity
                    })
                    layer_densities = [
                        masks[name].sum().item() / masks[name].numel()
                        for name in masks
                    ]
                    wandb.log({
                        "round": t_round,
                        f"client_{cid}/layer_density": wandb.Histogram(layer_densities)
                    })

        # aggregate ang log global metrics
        global_weights = average_weights_fedavg(local_weights, num_samples_list)
        log_global_weight_diff(prev_global_weights, global_weights, t_round)
        global_model.load_state_dict(global_weights)

        # model evaluation
        metrics = evaluate(global_model, test_loader)
        log_global_metrics(metrics, t_round)
        log_round_info(
            t_round,
            len(selected_clients),
            sum(len(client_datasets[cid]) for cid in selected_clients)
        )

        if t_round % 5 == 0 or t_round == config.ROUNDS:
            checkpoint_dir = config.OUT_CHECKPOINT_PATH
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"fl_model_round_{t_round}.pth")
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        print(f"Round {t_round} complete — Global loss: {metrics['global_loss']:.4f}, metrics: {metrics}\n")

    print(f"\n{'=' * 10} Training Completed {'=' * 10}")
    return global_model


if __name__ == "__main__":
    main()
