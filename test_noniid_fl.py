import os
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.dino_ViT_b16 import DINO_ViT
from fl_core.client import local_train
from fl_core.server import average_weights_fedavg
from data.prepare_data_fl import get_client_datasets, get_test_loader
from project_utils.metrics import get_metrics


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


def run_all_experiments():
    Nc_values = [1, 5, 10, 50]
    J_values = [4, 8, 16]

    for nc in Nc_values:
        for j in J_values:
            print("=" * 80)
            print(f"▶️  Start run: IID=False | Nc={nc} | J={j} | Rounds={800 // j}")
            print("=" * 80)

            config = {
                "IID": False,
                "LOCAL_EPOCHS": j,
                "NC": nc,
                "ROUNDS": 800 // j,
                "CLIENT_FRACTION": 0.1,
                "NUM_CLIENTS": 100,
                "BATCH_SIZE": 32,
                "LR": 0.001,
                "LR_DECAY": 1.0,
                "WARMUP_EPOCHS": 0,
                "FINETUNE_METHOD": "dense",
                "seed": 42,
            }

            np.random.seed(config['seed'])
            torch.manual_seed(config['seed'])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  Using CPU")

            model = DINO_ViT(num_classes=100, pretrained=True)
            test_loader = get_test_loader(batch_size=config['BATCH_SIZE'])
            client_datasets = get_client_datasets(False, config['NUM_CLIENTS'], config['NC'], config['seed'])

            for t_round in range(1, config['ROUNDS'] + 1):
                print(f"\n--- Round {t_round} ---")
                m = max(int(config['CLIENT_FRACTION'] * config['NUM_CLIENTS']), 1)
                selected_clients = np.random.choice(config['NUM_CLIENTS'], m, replace=False)

                local_weights, num_samples_list = [], []
                lr_round = config['LR'] * (config['LR_DECAY'] ** (t_round - 1))

                for cid in selected_clients:
                    print(f"Training client -> {cid}")
                    local_model = copy.deepcopy(model)
                    loader = DataLoader(client_datasets[cid], batch_size=config['BATCH_SIZE'], shuffle=True)
                    w, _, _ = local_train(local_model, loader, epochs=config['LOCAL_EPOCHS'],
                                          lr=lr_round, device=device, warmup_epochs=config['WARMUP_EPOCHS'])
                    local_weights.append(w)
                    num_samples_list.append(len(client_datasets[cid]))

                global_weights = average_weights_fedavg(local_weights, num_samples_list)
                model.load_state_dict(global_weights)
                metrics = evaluate(model, test_loader)
                print(f"Round {t_round} complete — Global loss: {metrics['global_loss']:.4f}, Top-1 Acc: {metrics['top_1_accuracy']:.4f}")

    print("\n✅ Tutti gli esperimenti Non-IID completati!")


if __name__ == "__main__":
    run_all_experiments()
