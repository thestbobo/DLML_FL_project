import torch
import wandb
import torch.nn as nn
from data.prepare_data import get_cifar100_loaders
import yaml
from tqdm import tqdm
from pathlib import Path
from models.dino_ViT_b16 import DINO_ViT
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# CUDA status
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU NOT available, using CPU!")

# Load YAML config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# WANDB LOGS SETUP
wandb.init(project="CIFAR-100_centralized", config=config)

# DATA
DATA_DIR = Path("./data")
train_loader, val_loader, test_loader = get_cifar100_loaders(
    config["val_split"], 
    config["batch_size"], 
    config["num_workers"]
)

# Model definition with DINO ViT-S/16
model = DINO_ViT(
    num_classes=100,
    pretrained=True,
    unfreeze_last_block=config.get("unfreeze_last_block", False)
).to(device)

# Loss function with label smoothing
criterion = nn.CrossEntropyLoss(
    label_smoothing=config.get("label_smoothing", 0.1)
).to(device)

# Optimizer - only trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    trainable_params,
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
    momentum=config["momentum"]
)

# Scheduler with warmup
warmup_epochs = config.get("warmup_epochs", 10)
warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=0.01, 
    total_iters=warmup_epochs
)
cosine_scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=config["epochs"] - warmup_epochs
)
scheduler = SequentialLR(
    optimizer, 
    schedulers=[warmup_scheduler, cosine_scheduler], 
    milestones=[warmup_epochs]
)

def train_one_epoch(model, dataloader, optimizer, criterion, device, curr_epoch, verbose=False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    loader = tqdm(dataloader, desc=f"Train Epoch {curr_epoch}", leave=False) if verbose else dataloader

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

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
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# Training loop
best_val_accuracy = 0.0
Path("checkpoints").mkdir(exist_ok=True)

for epoch in range(config["epochs"]):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch + 1, verbose=True
    )
    val_loss, val_acc = validate(
        model, val_loader, criterion, device, verbose=True
    )
    scheduler.step()

    print(
        f"Epoch {epoch + 1}/{config['epochs']} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
    )

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr": scheduler.get_last_lr()[0]
    })

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_accuracy': val_acc,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'train_loss': train_loss,
        }, f"checkpoints/checkpoint_{epoch}.pth")
        print(f'Checkpoint saved with Val Acc={val_acc:.2f}%')

    # Save best model
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val_accuracy': val_acc,
            'best_val_loss': val_loss,
            'best_train_accuracy': train_acc,
            'best_train_loss': train_loss,
        }, 'checkpoints/best_model.pth')
        print(f'Best model saved with Val Acc={best_val_accuracy:.2f}%')

# Save final model
torch.save(model.state_dict(), "checkpoints/dino_vit_final.pth")
print("Training completed!")