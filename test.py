import os
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
    # checkpoint_name = "best_cent_ckpt_dense_epoch_49.pth"
    checkpoint_path = os.path.join("checkpoints", "12g556mn")   # two separate folders for sparse and dense
    best_acc, best_loss = 0.0, 100.0
    best_loss_checkpoint = None
    best_acc_checkpoint = None

    for checkpoint_name in sorted(os.listdir(checkpoint_path)):
        checkpoint_p = os.path.join(checkpoint_path, checkpoint_name)
        checkpoint = torch.load(checkpoint_p, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint_name[0:4] == "best":
            metrics = checkpoint.get('best_val_metrics', {})
            val_loss = checkpoint.get('best_val_loss', 0.0)
        else:
            metrics = checkpoint.get('val_metrics', {})
            val_loss = checkpoint.get('val_loss', 0.0)

        print(f"\nLoaded model from epoch {checkpoint['epoch']} with validation metrics:")
        print(f"\tTop-1 Accuracy: {metrics.get('top_1_accuracy', 0.0) * 100:.2f}%")
        print(f"\tTop-5 Accuracy: {metrics.get('top_5_accuracy', 0.0) * 100:.2f}%")
        print(f"\tF1-Score: {metrics.get('f1_score', 0.0) * 100:.2f}%")
        print(f"\tLoss: {val_loss:.2f}")

        # Loss function
        criterion = nn.CrossEntropyLoss().to(device)

        # # Optional: initialize WandB run for test logging
        # wandb.init(project="CIFAR-100_centralized", name="final_test", config=config)

        # Run test
        test_loss, test_acc = test(model, test_loader, criterion, device, verbose=True)

        if best_loss > test_loss:
            best_loss = test_loss
            best_loss_checkpoint = checkpoint_name
        if best_acc < test_acc:
            best_acc = test_acc
            best_acc_checkpoint = checkpoint_name

        # # Log to WandB
        # wandb.log({
        #     "test_loss": test_loss,
        #     "test_accuracy": test_acc
        # })
        #
        # wandb.finish()


if __name__ == "__main__":
    main()
