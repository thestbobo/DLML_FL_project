import wandb
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from models.dino_ViT_b16 import DINO_ViT
from data.prepare_data import get_cifar100_loaders
from project_utils.metrics import get_metrics

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, curr_epoch, verbose=False):
    model.train()
    running_loss, total = 0.0, 0
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

        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
        all_outputs.append(outputs)
        all_labels.append(labels)

    avg_loss = running_loss / total
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = get_metrics(all_outputs, all_labels)

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
    metrics = get_metrics(all_outputs, all_labels)

    return avg_loss, metrics


def main():
    # cuda status
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # load YAML config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # WANDB LOGS SETUP
    wandb.init(project="CIFAR-100_centralized", config=config)

    # DATA
    DATA_DIR = Path("./data")
    train_loader, val_loader, test_loader = get_cifar100_loaders(config["val_split"], config["batch_size"], 0)

    # create validation split

    # model definition
    model = DINO_ViT().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=config["t_max"])

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"] - 5)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])

    best_val_accuracy = 0.0

    for epoch in range(config["epochs"]):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch + 1, verbose=True)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, verbose=True)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Train Metrics: {train_metrics} | "
            f"Val Loss: {val_loss:.4f} | Val Metrics: {val_metrics}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        # saves model, optimizer, scheduler along with current values every 5 epochs
        if epoch % 5 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_metrics': val_metrics,
                        'val_loss': val_loss,
                        'train_metrics': train_metrics,
                        'train_loss': train_loss},
                       f"checkpoints/checkpoint_{epoch}.pth")
            print(f'Checkpoint saved with Val Metrics={val_metrics}')

        # saves the best model along with current values
        if val_metrics["top_1_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["top_1_accuracy"]
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_val_accuracy': val_metrics,
                        'best_val_loss': val_loss,
                        'best_train_accuracy': train_metrics,
                        'best_train_loss': train_loss},
                       'best_model.pth')
            print(f'Best model saved with Val Top-1 Accuracy={best_val_accuracy:.2f}%')

    torch.save(model.state_dict(), f"checkpoints/dino_vit_final.pt")

if __name__ == "__main__":
    main()
