import torch
import wandb
import torch.nn as nn
from data.prepare_data import get_cifar100_loaders
import yaml
from tqdm import tqdm
from pathlib import Path
from models.dino_ViT_b16 import DINO_ViT
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# cuda status
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU avaiable: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU NOT avaiable, using CPU!")


# load YAML config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# WANDB LOGS SETUP
wandb.init(project="CIFAR-100_centralized", config=config)

# DATA
DATA_DIR = Path("./data")
train_loader, val_loader, test_loader = get_cifar100_loaders(config["val_split"], config["batch_size"], config["num_workers"])

# create validation split

# model definition
dino = DINO_ViT()
model = dino.model
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(dino.classifier.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], momentum=config["momentum"])
# scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=config["t_max"])

warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"] - 5)

scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])


def train_one_epoch(model, dataloader, optimizer, criterion, device, curr_epoch, verbose=False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    loader = tqdm(train_loader, desc=f"Train Epoch {curr_epoch}", leave=False) if verbose else dataloader

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
        correct += torch.sum(predicted.eq(labels)).item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, verbose=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    loader = tqdm(val_loader, desc="Validating", leave=False) if verbose else val_loader

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


best_val_accuracy = 0.0

for epoch in range(config["epochs"]):
  train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch + 1, verbose=True)
  val_loss, val_acc = validate(model, val_loader, criterion, device, verbose=True)
  scheduler.step()

  print(
      f"Epoch {epoch + 1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

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
                  f"checkpoint_{epoch}.pth")
      print(f'Checkpoint saved with Acc={train_acc:.2f}%')

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
      print(f'Best model saved with Acc={best_val_accuracy:.2f}%')

torch.save(model.state_dict(), f"checkpoints/dino_vit_final.pt")