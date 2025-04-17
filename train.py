import wandb
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from models.dino_ViT_b16 import DINO_ViT
from data.prepare_data import get_cifar100_loaders

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, curr_epoch, verbose=False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

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
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += torch.sum(predicted.eq(labels)).item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, verbose=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    loader = tqdm(dataloader, desc="Validating", leave=False) if verbose else dataloader

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted.eq(labels)).item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy



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
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch + 1, verbose=True)
        val_loss, val_acc = validate(model, val_loader, criterion, device, verbose=True)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Acc: {val_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # saves model, optimizer, scheduler along with current values every 5 epochs
        if epoch % 5 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_accuracy': val_acc,
                        'val_loss': val_loss,
                        'train_accuracy': train_acc,
                        'train_loss': train_loss},
                       f"checkpoints/checkpoint_{epoch}.pth")
            print(f'Checkpoint saved with Acc={train_acc*100:.2f}%')

        # saves the best model along with current values
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_val_accuracy': val_acc,
                        'best_val_loss': val_loss,
                        'best_train_accuracy': train_acc,
                        'best_train_loss': train_loss},
                       'best_model.pth')
            print(f'Best model saved with Acc={best_val_accuracy*100:.2f}%')

    torch.save(model.state_dict(), f"checkpoints/dino_vit_final.pt")

if __name__ == "__main__":
    main()
