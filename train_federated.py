import os
import numpy as np
import torch
import yaml
import wandb
import random
from torch.utils.data import DataLoader
from models.dino_ViT_b16 import DINO_ViT
from data.prepare_data_fl import load_cifar100, split_iid, split_noniid
from fl_core.client import local_train
from fl_core.server import average_weights_fedavg
from data.prepare_data import get_test_transforms  # For test set
from torchvision.datasets import CIFAR100
from project_utils.metrics import get_metrics

# -------------------- CONFIG --------------------

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)
# ------------------------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


wandb.init(
    project="Federated-DINO-ViT",
    config={
        "model": "DINO ViT-S/16",
        "dataset": "CIFAR-100",
        "num_clients": config['NUM_CLIENTS'],
        "client_fraction": config["CLIENT_FRACTION"],
        "local_epochs": config["LOCAL_EPOCHS"],
        "batch_size": config["BATCH_SIZE"],
        "lr": config["LR"],
        "rounds": config["ROUNDS"],
        "iid": config["IID"],
        "nc": config["NC"] if not config["IID"] else None
    }
)


set_seed(config.seed)


def get_client_datasets():
    full_dataset = load_cifar100()

    if config['IID']:
        return split_iid(full_dataset, config["NUM_CLIENTS"])
    else:
        return split_noniid(full_dataset, config["NUM_CLIENTS"], nc=config['NC'], seed=config['SEED'])


def get_test_loader():
    test_transform = get_test_transforms()
    test_data = CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    return DataLoader(test_data, batch_size=config['BATCH_SIZE'], shuffle=False)


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


def train_federated():
    # cuda status
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    mode = "IID" if config["IID"] else f"Non-IID Nc={config['NC']}"
    print(f"{'=' * 10} Federated Training Start ({mode}) {'=' * 10}")

    # Load data
    client_datasets = get_client_datasets()
    test_loader = get_test_loader()

    # Initialize global model
    global_model = DINO_ViT(num_classes=100, pretrained=True)
    global_weights = global_model.state_dict()

    for round in range(1, config['ROUNDS'] + 1):
        print(f"\n--- Round {round} ---")

        # Sample clients
        m = max(int(config['CLIENT_FRACTION'] * config["NUM_CLIENTS"]), 1)
        selected_clients = np.random.choice(config["NUM_CLIENTS"], m, replace=False)

        local_weights = []
        num_samples_list = []
        for client_id in selected_clients:
            local_model = DINO_ViT(num_classes=100, pretrained=False)
            local_model.load_state_dict(global_weights)

            client_data = DataLoader(client_datasets[client_id], batch_size=config['BATCH_SIZE'], shuffle=True)
            updated_weights = local_train(local_model, client_data, config['LOCAL_EPOCHS'], config['LR'], device)
            local_weights.append(updated_weights)
            num_samples_list.append(len(client_datasets[client_id]))

        # Average updates
        global_weights = average_weights_fedavg(local_weights, num_samples_list)
        global_model.load_state_dict(global_weights)

        # Evaluate global model
        metrics = evaluate(global_model, test_loader)

        # log round results
        wandb.log({
            "round": round,
            **{f"global_test_{k}": v for k, v in metrics.items()},
            "clients_participated": len(selected_clients),
            "samples_used": sum(len(client_datasets[cid]) for cid in selected_clients)
        })

        # Save model checkpoint every 10 rounds (or customize)
        if round % 5 == 0 or round == config["ROUNDS"]:
            checkpoint_dir = "/content/DLML_FL_project/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"fl_model_round_{round}.pth")
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        print(f"Round {round}: Global Test Metrics = {metrics}")

    print(f"\n{'=' * 10} Training Completed {'=' * 10}")
    return global_model


if __name__ == "__main__":
    train_federated()
