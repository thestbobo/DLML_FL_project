import yaml
import torch
import wandb
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from models.dino_ViT_b16 import DINO_ViT
from data.prepare_data import get_cifar100_loaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(model, dataloader, criterion, device, verbose=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    loader = tqdm(dataloader, desc="Testing", leave=False) if verbose else dataloader

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
    print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy*100:.2f}%")
    return avg_loss, accuracy


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load config
    with open("config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    # Load data
    _, _, test_loader = get_cifar100_loaders(config["val_split"], config["batch_size"], config["num_workers"])

    # Init model
    model = DINO_ViT().to(device)

    # Load checkpoint
    checkpoint_name = "centralized_checkpoint_epoch_50.pth"
    checkpoint_path = "checkpoints\\sparse\\" + checkpoint_name   # two separate folders for sparse and no_sparse
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if checkpoint_name[0:4] == "best":
        val_acc = checkpoint.get('best_val_metrics', {}).get('top_1_accuracy', 0.0)
    else:
        val_acc = checkpoint.get('val_metrics', {}).get('top_1_accuracy', 0.0)
    print(f"Loaded model from epoch {checkpoint['epoch']} with val acc: {val_acc*100:.2f}%")

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # # Optional: initialize WandB run for test logging
    # wandb.init(project="CIFAR-100_centralized", name="final_test", config=config)

    # Run test
    test_loss, test_acc = test(model, test_loader, criterion, device, verbose=True)

    # # Log to WandB
    # wandb.log({
    #     "test_loss": test_loss,
    #     "test_accuracy": test_acc
    # })
    #
    # wandb.finish()


if __name__ == "__main__":
    main()
