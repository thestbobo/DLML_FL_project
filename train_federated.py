import os
import copy
import numpy as np
import torch
import yaml
import wandb
from torch.utils.data import DataLoader, ConcatDataset, Subset

from model_editing.TaLoS import compute_fisher_scores, calibrate_mask
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
    log_client_metrics
)


def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

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
    # load config
    with open("config/config.yaml") as f:
        default_config = yaml.safe_load(f)

    wandb.init(project="Federated-DINO-ViT", config=default_config)
    config = wandb.config

    # seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # check for cached mask
    if config.MASKS_DIR:
        # User has provided a path → we trust it contains mask_global.pt
        masks_root = config.MASKS_DIR
        os.makedirs(masks_root, exist_ok=True)
        print(f">>> Loading precomputed mask from: {masks_root}")
        need_to_compute_mask = False
    else:
        # No path given → compute a new mask this run and save under ./masks_run
        masks_root = "./masks_run"
        os.makedirs(masks_root, exist_ok=True)
        print(f">>> No MASKS_DIR set; will compute new mask and store under: {masks_root}")
        need_to_compute_mask = True

        # Filenames for the shared Fisher scores and mask
    global_fisher_file = os.path.join(masks_root, "fisher_global.pt")
    global_mask_file = os.path.join(masks_root, "mask_global.pt")

    mode = "IID" if config.IID else f"Non-IID Nc={config.NC}"
    print(f"========== Federated Training Start ({mode}) ==========")

    # (optional) resume checkpoint
    starting_round = 0
    best_test_accuracy = 0.0
    ckpt_path = config.get("checkpoint_path", "")

    global_model = DINO_ViT(num_classes=100, pretrained=True)

    print("[INFO] Loading from checkpoint:", config.CHECKPOINT_PATH)
    print("[INFO] Exists?", os.path.exists(config.CHECKPOINT_PATH))


    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path} …")
        ckpt = torch.load(ckpt_path, map_location=device)
        method = ckpt.get("finetuning_method", "").lower()
        strict = (method != "lora")
        global_model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        starting_round = ckpt.get("round", 0)
        best_test_accuracy = ckpt.get("test_metrics", {}).get("top_1_accuracy", 0.0)
        print(f"Resumed from round {starting_round} with best test@1 = {best_test_accuracy * 100:.2f}%")
    else:
        print("No valid checkpoint found; starting from scratch.")

    # prepare data
    client_datasets = get_client_datasets(config.IID, config.NUM_CLIENTS, config.seed)
    test_loader = get_test_loader(batch_size=config.BATCH_SIZE)

    if need_to_compute_mask:
        num_calib_clients = min(5, config.NUM_CLIENTS)
        per_client_calib_count = int(len(client_datasets[0]) * config.CALIBRATION_SPLIT)

        subsets = []
        for cid in range(num_calib_clients):
            full_dataset = client_datasets[cid]
            indices = list(range(per_client_calib_count))
            subsets.append(Subset(full_dataset, indices))

        calibration_dataset = ConcatDataset(subsets)
        calib_loader = DataLoader(
            calibration_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
    else:
        calib_loader = None

    print("\n>>> Preparing shared Fisher+mask ...")
    if need_to_compute_mask:
        # 7.1) Compute Fisher scores once on calib_loader
        dummy = copy.deepcopy(global_model).to(device)
        dummy_criterion = torch.nn.CrossEntropyLoss()
        fisher_scores = compute_fisher_scores(dummy, calib_loader, dummy_criterion, device)
        torch.save(fisher_scores, global_fisher_file)
        del dummy  # free GPU

        # 7.2) One-shot prune of `target_sparsity` fraction of ALL weights
        shared_masks = calibrate_mask(
            fisher_scores,
            target_sparsity=config.TALOS_TARGET_SPARSITY,
            rounds=1
        )
        torch.save(shared_masks, global_mask_file)
    else:
        # Load from existing folder
        fisher_scores = torch.load(global_fisher_file, map_location=device)
        shared_masks = torch.load(global_mask_file, map_location=device)

    print(">>> Shared mask ready (loaded from or saved to) →", masks_root)

    # get LR info for scheduler configuration
    base_lr = config.LR
    decay = config.LR_DECAY
    warmup_eps = config.WARMUP_EPOCHS

    # federated loop
    for t_round in range(starting_round + 1, config.ROUNDS + 1):
        print(f"\n--- Round {t_round} ---")

        m = max(int(config.CLIENT_FRACTION * config.NUM_CLIENTS), 1)
        selected_clients = np.random.choice(config.NUM_CLIENTS, m, replace=False)

        # logging class distribution for all selected clients every k rounds
        if t_round % 5 == 0:
            log_aggregated_class_distribution(client_datasets, selected_clients, t_round)

        prev_global_weights = global_model.state_dict()
        local_weights, num_samples_list = [], []
        client_to_log = np.random.choice(selected_clients)   # logging local metrics for one random active client

        # compute per round LR decay
        lr_round = base_lr * (decay ** (t_round - 1))

        for cid in selected_clients:
            print(f"Training client -> {cid}")
            # DEBUG
            cnt = 0
            for _ in DataLoader(client_datasets[cid], batch_size=1):
                cnt += 1
            print(f"  Client {cid} dataset size: {cnt}")

            local_model = copy.deepcopy(global_model)

            loader = DataLoader(client_datasets[cid],
                                batch_size=config.BATCH_SIZE,
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True)

            # DEBUG
            initial_weights = copy.deepcopy(local_model.state_dict())
            method = config.FINETUNE_METHOD.lower()

            if method == "dense":
                w, avg_loss, acc = local_train(local_model,
                                               loader,
                                               epochs=config.LOCAL_EPOCHS,
                                               lr=lr_round,
                                               device=device,
                                               warmup_epochs=warmup_eps)
                sparsity = None
                masks = None

            elif method == "talos":
                # ─── Use the PRECOMPUTED global mask (same for every client) ───
                # We pass `global_masks` and skip any recalculation inside local_train_talos.
                w, avg_loss, acc, sparsity, masks = local_train_talos(
                    local_model,
                    loader,
                    epochs=config.LOCAL_EPOCHS,
                    lr=lr_round,
                    device=device,
                    target_sparsity=config.TALOS_TARGET_SPARSITY,
                    prune_rounds=config.TALOS_PRUNE_ROUNDS,
                    warmup_epochs=warmup_eps,
                    masks_dir=config.MASKS_DIR,  # client_id is unused when loading global mask
                    global_masks=shared_masks  # pass the one global mask
                )
            else:
                raise ValueError(f"Unknown FINETUNE_METHOD '{method}'")

            # DEBUG: Compare initial vs final weights
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

            # logging single selected client
            if cid == client_to_log:
                log_client_metrics(cid, avg_loss, acc, t_round)
                if sparsity is not None:
                    # log global sparsity
                    wandb.log({
                        "round": t_round,
                        f"client_{cid}/sparsity": sparsity
                    })
                    # log layer-wise density histogram
                    layer_densities = [
                        masks[name].sum().item() / masks[name].numel() for name in masks
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
