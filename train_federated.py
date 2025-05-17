import os
import numpy as np
import torch
import yaml
import wandb
from torch.utils.data import DataLoader
from models.dino_ViT_b16 import DINO_ViT
from fl_core.client import local_train, local_train_talos
from fl_core.server import average_weights_fedavg
from data.prepare_data_fl import get_client_datasets, get_test_loader
from project_utils.metrics import get_metrics


def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            all_outputs.append(outputs)
            all_labels.append(y)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = get_metrics(all_outputs, all_labels)  # Calcola le metriche

    return metrics


def main():
    # LOAD CONFIG
    with open("config/config.yaml") as f:
        default_config = yaml.safe_load(f)

    # WandB CONFIG
    wandb.init(
        project="Federated-DINO-ViT",
        config=default_config
    )

    config = wandb.config
    config.NC = config.NC if not config.IID else None
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # GPU CHECK
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    mode = "IID" if config.IID else f"Non-IID Nc={config.NC}"
    print(f"{'=' * 10} Federated Training Start ({mode}) {'=' * 10}")

    # CHECKPOINT LOADING (optional, to be enabled in config)
    starting_round = 0
    best_test_accuracy = 0.0

    ckpt_path = config.get("checkpoint_path", "")
    global_model = DINO_ViT(num_classes=100, pretrained=True)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path} …")
        ckpt = torch.load(ckpt_path, map_location=device)

        method = ckpt.get("finetuning_method", "").lower()
        strict = (method != "lora")  # example for other methods
        global_model.load_state_dict(ckpt["model_state_dict"], strict=strict)

        starting_round = ckpt.get("round", 0)
        best_test_accuracy = ckpt.get("test_metrics", {}) \
            .get("top_1_accuracy", 0.0)

        print(f"Resumed from round {starting_round} "
              f"with best test@1 = {best_test_accuracy * 100:.2f}%")
    else:
        print("No valid checkpoint found; starting from scratch.")

    mode = "IID" if config.IID else f"Non-IID Nc={config.NC}"
    print(f"{'=' * 10} Federated Training Start ({mode}) {'=' * 10}")

    # DATA PREPARATION
    client_datasets = get_client_datasets(config.IID, config.NUM_CLIENTS, config.seed)
    test_loader = get_test_loader(batch_size=config.BATCH_SIZE)

    # Initialize global model
    global_weights = global_model.state_dict()

    for t_round in range(starting_round + 1, config.ROUNDS + 1):
        print(f"\n--- Round {t_round} ---")

        # sample clients
        m = max(int(config.CLIENT_FRACTION * config.NUM_CLIENTS), 1)
        selected_clients = np.random.choice(config.NUM_CLIENTS, m, replace=False)

        local_weights, num_samples_list = [], []
        for cid in selected_clients:
            # init local model
            local_model = DINO_ViT(num_classes=100, pretrained=False)
            local_model.load_state_dict(global_weights)

            loader = DataLoader(client_datasets[cid],
                                batch_size=config.BATCH_SIZE,
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True)

            # dispatch fine-tuning method
            method = config.FINETUNE_METHOD.lower()
            if method == "dense":
                w = local_train(
                    local_model, loader,
                    epochs=config.LOCAL_EPOCHS,
                    lr=config.LR,
                    device=device
                )
            elif method == "talos":
                w = local_train_talos(
                    local_model, loader,
                    epochs=config.LOCAL_EPOCHS,
                    lr=config.LR,
                    device=device,
                    target_sparsity=config.TALOS_TARGET_SPARSITY,
                    prune_rounds=config.TALOS_PRUNE_ROUNDS,
                    fisher_loader=None  # defaults to training loader
                )
            else:
                raise ValueError(f"Unknown FINETUNE_METHOD '{method}'")

            local_weights.append(w)
            num_samples_list.append(len(client_datasets[cid]))

        # AGGREGATION OF THE CLIENTS UPDATES-> FedAvg
        global_weights = average_weights_fedavg(local_weights, num_samples_list)
        global_model.load_state_dict(global_weights)

        # VALIDATION OF GLOBAL MODEL
        metrics = evaluate(global_model, test_loader)

        # log round results
        wandb.log({
            "round": t_round,
            **{f"global_test_{k}": v for k, v in metrics.items()},
            "clients_participated": len(selected_clients),
            "samples_used": sum(len(client_datasets[cid]) for cid in selected_clients)
        })

        # CHECKPOINT SAVING (every 5 rounds)
        if t_round % 5 == 0 or t_round == config.ROUNDS:
            checkpoint_dir = config.OUT_CHECKPOINT_PATH
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"fl_model_round_{t_round}.pth")
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        print(f"Round {t_round}: Global Test Metrics = {metrics}")

    print(f"\n{'=' * 10} Training Completed {'=' * 10}")
    return global_model


if __name__ == "__main__":
    main()
