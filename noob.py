import torch

checkpoint = torch.load("/Users/gabrieleadorni/Downloads/checkpoint_15.pth", map_location="cpu")

print(checkpoint.keys())
print(checkpoint["train_metrics"])
print(checkpoint["val_loss"])
print(checkpoint["train_loss"])
