import re
import os
import copy
import time
import yaml
import torch
import wandb
import numpy as np

from torch.utils.data import DataLoader, ConcatDataset

from model_editing.TaLoS import compute_fisher_scores, calibrate_mask_global, calibrate_mask_layerwise_qk, calibrate_mask_layerwise_qk_ls
from models.dino_ViT_s16 import DINO_ViT

from fl_core.client import local_train, local_train_talos
from fl_core.server import FedAlignAvg

from data.prepare_data_fl import get_client_datasets, get_test_loader

from project_utils.metrics import get_metrics
from project_utils.federated_metrics import (
    log_global_weight_diff,
    log_aggregated_class_distribution,
    log_round_info,
    log_global_metrics,
    log_client_metrics)

from representations.representations_manager import get_intermediate_representation, save_representations


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


def load_checkpoint(model, optimizer, scheduler, path, config):
    if not os.path.exists(path):
        print("[WARN] No checkpoint found.")
        return 0  # start from scratch

    print(f"[INFO] Loading checkpoint from {path} …")
    ckpt = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if "model_state_dict" not in ckpt:
        print("[WARN] Checkpoint is a pure state_dict. Loading only weights.")
        model.load_state_dict(ckpt, strict=False)
        match = re.search(r"round_(\\d+)", path)
        return int(match.group(1)) if match else 0

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    seed = ckpt.get("seed", config.seed)
    np_seed = ckpt.get("np_seed", None)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if np_seed:
        np.random.set_state(np_seed)

    print(f"[INFO] Resumed from round {ckpt['round']} with top-1 acc = {ckpt.get('test_metrics', {}).get('top_1_accuracy', 0.0):.2%}")
    return ckpt["round"]



def main():
    # load config / init WandB
    with open("config/config.yaml", encoding="utf-8") as f:
        default_config = yaml.safe_load(f)

    if default_config["run_id"] != "" and default_config["run_id"] is not None:
        wandb.init(project="Federated-DINO-ViT", id=default_config["run_id"], config=default_config)
    else:
        wandb.init(project="Federated-DINO-ViT", config=default_config)

    config = wandb.config

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    method = config.FINETUNE_METHOD.lower()

    # manage mask cache paths
    if method == "talos":
        if method == "talos":
            if config.LOAD_MASK:
                global_mask_file = config.LOAD_MASK
                masks_root = os.path.dirname(global_mask_file)
                os.makedirs(masks_root, exist_ok=True)
                print(f">>> Loading precomputed mask from: {global_mask_file}")
                need_to_compute_mask = False
            else:
                masks_root = config.MASKS_DIR
                os.makedirs(masks_root, exist_ok=True)
                print(f">>> No LOAD_MASK set; computing new mask and storing under: {masks_root}")
                need_to_compute_mask = True
                global_mask_file = os.path.join(masks_root, "mask_global.pt")

            global_fisher_file = os.path.join(masks_root, "fisher_global.pt")
    else:
        masks_root = None
        need_to_compute_mask = False
        global_fisher_file = None
        global_mask_file = None

    mode = "IID" if config.IID else f"Non-IID Nc={config.NC}"
    print(f"========== Federated Training Start ({mode}) ==========")

    ckpt_path = config.CHECKPOINT_PATH

    # building model / loading existing one from config
    if method == "talos":
        global_model = DINO_ViT(num_classes=100, pretrained=True, frozen_backbone=False)
    else:
        global_model = DINO_ViT(num_classes=100, pretrained=True, frozen_backbone=True)

    print("[INFO] Loading from checkpoint:", ckpt_path)
    print("[INFO] Exists?", os.path.exists(ckpt_path))

    optimizer = None  # Definisci qui se usi ottimizzatore
    scheduler = None  # Idem per scheduler
    starting_round = load_checkpoint(global_model, optimizer, scheduler, ckpt_path, config)


    # data prep
    client_datasets = get_client_datasets(config.IID, config.NUM_CLIENTS, config.NC, config.downsample_frac, config.seed)
    test_loader = get_test_loader(batch_size=config.BATCH_SIZE)

    # talos branch
    if method == "talos":
        if need_to_compute_mask:
            print("\n>>> Building a calibration loader over the FULL CIFAR-100 training set …")
            # concatenate all client splits.
            """ Shuffle=True to reduce ordering bias and achieve a better gradient coverage"""
            full_train_dataset = ConcatDataset(client_datasets)
            fisher_loader = DataLoader(
                full_train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
        else:
            fisher_loader = None

        print("\n>>> Preparing shared Fisher + mask (TaLoS) …")
        if need_to_compute_mask:
            # track the time it takes to calculate the fisher scores
            fisher_timer_start = time.perf_counter()

            # Compute Fisher scores on the entire dataset
            dummy = copy.deepcopy(global_model).to(device)
            dummy_criterion = torch.nn.CrossEntropyLoss()

            fisher_scores = compute_fisher_scores(dummy, fisher_loader, dummy_criterion, device)

            fisher_timer_end = time.perf_counter()
            wandb.config.fisher_runtime_min = (fisher_timer_end - fisher_timer_start) / 60

            # ----DEBUG----
            print(">>> Number of entries in fisher_scores:", len(fisher_scores))
            # Print a few QKV entries:
            for name in sorted(fisher_scores):
                if "qkv.weight" in name or "qkv.bias" in name:
                    print("  FISHER[QKV] →", name, "mean=", fisher_scores[name].mean().item())

            print("\n>>> [DEBUG] Number of entries in fisher_scores:", len(fisher_scores))
            # Look for any key that contains "qkv"
            for name in sorted(fisher_scores.keys()):
                if "qkv.weight" in name or "qkv.bias" in name:
                    print("    FISHER[SCORE] key:", name,
                          "   mean=", fisher_scores[name].mean().item(),
                          "   max=", fisher_scores[name].max().item(),
                          "   min=", fisher_scores[name].min().item())
            # If you want to print just the first few keys overall:
            print(">>> [DEBUG] First 10 fisher_scores keys:", list(fisher_scores.keys())[:10], "\n")
            # --------------


            mask_timer_start = time.perf_counter()

            # Build a layer‐wise Q/K float mask (keep least sensitive)
            if config.TALOS_MASK_TYPE == "qk_ms":
                shared_masks = calibrate_mask_layerwise_qk(
                    dummy,
                    fisher_scores,
                    keep_ratio_per_block=(1.0 - config.TALOS_TARGET_SPARSITY),
                    target_qk_sparsity=config.TALOS_TARGET_SPARSITY,
                    max_rounds=config.TALOS_PRUNE_ROUNDS
                )

            # Build a layer‐wise Q/K float mask (keep most sensitive)
            elif config.TALOS_MASK_TYPE == "qk_ls":
                shared_masks = calibrate_mask_layerwise_qk_ls(
                    model=global_model,
                    fisher_scores=fisher_scores,
                    target_qk_sparsity=config.TALOS_TARGET_SPARSITY,
                    max_rounds=config.TALOS_PRUNE_ROUNDS
                )

            # Build a global mask (keep least sensitive)
            elif config.TALOS_MASK_TYPE == "global":
                shared_masks = calibrate_mask_global(
                    model=dummy,
                    calib_loader=fisher_loader,
                    criterion=dummy_criterion,
                    device=device,
                    target_sparsity=config.TALOS_TARGET_SPARSITY,
                    rounds=config.TALOS_PRUNE_ROUNDS,
                    random_fallback_frac=0.1,
                    seed=config.seed,
                    min_keep_frac=0.05,
                )
            else:
                raise ValueError("Unknown mask type:", config.TALOS_MASK_TYPE)

            mask_timer_end = time.perf_counter()
            wandb.config.mask_runtime_min = (mask_timer_start - mask_timer_end) / 60

            total = sum(m.numel() for m in shared_masks.values())
            kept = sum(int(m.sum().item()) for m in shared_masks.values())
            print(f"[DEBUG] GLOBAL MASK → kept {kept}/{total} ≈ {100 * kept / total:.1f}% of all params")

            torch.save(shared_masks, global_mask_file)
            del dummy, fisher_scores

        else:
            # load pre-computed mask
            shared_masks = torch.load(global_mask_file, map_location=device)

            # load pre-computed fisher scores (comment out if needed pls)
            # fisher_scores = torch.load(global_fisher_file, map_location=device)

            total = sum(m.numel() for m in shared_masks.values())
            kept = sum(int(m.sum().item()) for m in shared_masks.values())
            print(f"[DEBUG] LOADED GLOBAL MASK WITH → kept {kept}/{total} ≈ {100 * kept / total:.1f}% of all params")

        print(
            f">>> Shared mask ready ({'loaded from' if not need_to_compute_mask else 'computed and saved to'}) → {masks_root}")

    else:
        shared_masks = None
        print("\n>>> Skipping Fisher/mask preparation — dense training.")

    # lr.
    base_lr = config.LR
    decay = config.LR_DECAY
    warmup_eps = config.WARMUP_EPOCHS

    # representations extraction setup
    extract_every_n_rounds = config.REPRESENTATION_FREQ
    repr_layers = config.REPRESENTATION_LAYERS  # list like ['model.blocks.3', 'model.blocks.6']
    repr_path = config.REPRESENTATIONS_PATH

    # federated loop
    for t_round in range(starting_round + 1, config.ROUNDS + 1):
        print(f"\n--- Round {t_round} ---")

        # Select a subset of clients
        selected_clients = np.arange(config.NUM_CLIENTS)

        # Always grab a held-out batch for SVCCA probing
        probe_loader = get_test_loader(batch_size=config.BATCH_SIZE)
        x_probe, _ = next(iter(probe_loader))

        # Independently mark which clients to SAVE representations
        clients_to_extract = []
        if t_round % extract_every_n_rounds == 0:
            num_repr_clients = config.REPRESENTATION_CLIENTS_PER_ROUND
            clients_to_extract = list(np.random.choice(
                selected_clients,
                min(num_repr_clients, len(selected_clients)),
                replace=False
            ))


        # Log aggregated class distribution every 5 rounds
        if t_round % 5 == 0:
            log_aggregated_class_distribution(client_datasets, selected_clients, t_round)

        prev_global_weights = global_model.state_dict()
        local_weights, num_samples_list = [], []
        repr_list = []
        client_to_log = np.random.choice(selected_clients)  # pick one client for local‐metrics logging
        lr_round = base_lr * (decay ** (t_round - 1))

        for cid in selected_clients:
            print(f"Training client -> {cid}")

            cnt = sum(1 for _ in DataLoader(client_datasets[cid], batch_size=1))
            # DEBUG
            print(f"  Client {cid} dataset size: {cnt}")

            local_model = copy.deepcopy(global_model)
            loader = DataLoader(
                client_datasets[cid],
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )

            # extract representations every `extract_every_n_rounds`
            extract_fn = None
            if t_round % extract_every_n_rounds == 0 or t_round == config.ROUNDS:
                # compute class distribution for this client (optional)
                if cid in clients_to_extract:
                    label_counter = {}
                    for _, label in client_datasets[cid]:
                        label_counter[label] = label_counter.get(label, 0) + 1

                    def extract_fn(model_ref=local_model, client_id=cid, round_id=t_round):
                        reps = get_intermediate_representation(model_ref, x_probe, repr_layers, device)
                        save_representations(reps, repr_path, client_id, round_id, class_counts=label_counter)

                    print(f"[REPRESENTATIONS] Scheduled extraction for Client {cid} at Round {t_round}")
                    print(f"[REPRESENTATIONS] Client {cid} class distribution: {label_counter}")

            # Keep track of initial weights to log weight‐delta (L2)
            initial_weights = copy.deepcopy(local_model.state_dict())
            method = config.FINETUNE_METHOD.lower()

            if method == "dense":
                w, avg_loss, acc = local_train(
                    local_model,
                    loader,
                    local_steps=config.LOCAL_STEPS,
                    lr=lr_round,
                    device=device,
                    warmup_steps=warmup_eps * (cnt // config.BATCH_SIZE),
                    extract_repr_fn = extract_fn
                )
                sparsity = None
                masks = None

            elif method == "talos":
                w, avg_loss, acc, sparsity, masks = local_train_talos(
                    local_model,
                    loader,
                    local_steps=config.LOCAL_STEPS,
                    lr=lr_round,
                    device=device,
                    target_sparsity=config.TALOS_TARGET_SPARSITY,
                    prune_rounds=config.TALOS_PRUNE_ROUNDS,
                    masks_dir=masks_root,                               # pass the same root where we saved global_mask
                    global_masks=shared_masks,                          # force‐use the precomputed global mask
                    warmup_steps=warmup_eps * (cnt // config.BATCH_SIZE),
                    extract_repr_fn=extract_fn
                )

                # Compute QKV sparsity (for logging) by counting mask entries under "attn.qkv"
                qk_total = 0
                qk_kept = 0
                for name, mask_tensor in masks.items():
                    # look for the fused QKV weight & bias
                    if "attn.qkv.weight" in name:
                        D_out, D_in = mask_tensor.shape
                        chunk = D_out // 3
                        kept_q = mask_tensor[0:chunk, :].sum().item()
                        kept_k = mask_tensor[chunk:2 * chunk, :].sum().item()
                        qk_kept += (kept_q + kept_k)
                        qk_total += (2 * chunk * D_in)

                    if "attn.qkv.bias" in name:
                        D_out = mask_tensor.numel()
                        chunk = D_out // 3
                        kept_qb = mask_tensor[0:chunk].sum().item()
                        kept_kb = mask_tensor[chunk:2 * chunk].sum().item()
                        qk_kept += (kept_qb + kept_kb)
                        qk_total += (2 * chunk)

                # Now compute sparsity = fraction pruned = 1 − (kept / total)
                qk_sparsity = 1.0 - (qk_kept / qk_total) if qk_total > 0 else 0.0

                print(f"  Client {cid} sparsity={100 * qk_sparsity:.2f}%")
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

            # extract client repr H_i on the same x_probe
            with torch.no_grad():
                reps_i = get_intermediate_representation(local_model, x_probe, repr_layers, device)

            # flatten each to (B, features)
            flattened = []
            for name in repr_layers:
                t = reps_i[name]
                t_flat = t.reshape(t.shape[0], -1)
                flattened.append(t_flat)

            H_i = torch.cat(flattened, dim=1)
            repr_list.append(H_i.cpu())

            # Log this client’s metrics when it was randomly selected
            if cid == client_to_log:
                log_client_metrics(cid, avg_loss, acc, t_round)
                if sparsity is not None:
                    wandb.log({
                        "round": t_round,
                        f"client_{cid}/qk_sparsity": sparsity
                    })


        with torch.no_grad():
            reps_glob = get_intermediate_representation(global_model, x_probe, repr_layers, device)

        flattened_glob = [reps_glob[name].reshape(reps_glob[name].shape[0], -1)
                          for name in repr_layers]

        H_glob = torch.cat(flattened_glob, dim=1)
        H_glob = H_glob.cpu()

        # drift-aware aggregation
        global_weights = FedAlignAvg(
            local_weights,
            repr_list,
            num_samples_list,
            H_glob,
            k=config.SVCCA_K,
            pca_dim=config.SVCCA_PCA_DIM,
            max_samples=config.SVCCA_MAX_SAMPLES)
        global_model.load_state_dict(global_weights)

        # log global metrics
        log_global_weight_diff(prev_global_weights, global_weights, t_round)

        # model evaluation
        metrics = evaluate(global_model, test_loader)
        log_global_metrics(metrics, t_round)
        log_round_info(
            t_round,
            len(selected_clients),
            sum(len(client_datasets[cid]) for cid in selected_clients)
        )

        if t_round % extract_every_n_rounds  == 0 or t_round == config.ROUNDS:
            checkpoint_dir = config.OUT_CHECKPOINT_PATH
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"fl_model_round_{t_round}.pth")
            torch.save(global_model.state_dict(), checkpoint_path)
            global_representation = get_intermediate_representation(global_model, x_probe, repr_layers, device)
            save_representations(global_representation, repr_path, client_id="global", round_num=t_round)
            print(f"Saved checkpoint and global representation: {checkpoint_path}")

            # save each selected client's weights alongside global
            for cid, w in zip(selected_clients, local_weights):
                client_ckpt = os.path.join(
                    checkpoint_dir, f"client{cid}_round_{t_round}.pth"
                )
                torch.save(w, client_ckpt)
                print(f"[CKPT] Saved client {cid} weights → {client_ckpt}")

        print(f"Round {t_round} complete — Global loss: {metrics['global_loss']:.4f}, metrics: {metrics}\n")

    print(f"\n{'=' * 10} Training Completed {'=' * 10}")
    return global_model



if __name__ == "__main__":
    main()