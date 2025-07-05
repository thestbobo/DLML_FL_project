# print_dino_vit_params.py

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from models.dino_ViT_b16 import DINO_ViT

def main():
    # pretrained=False so we don’t try to download any weights
    model = DINO_ViT(num_classes=100, pretrained=False)
    model.eval()

    print("=== All parameter names in DINO_ViT (random init) ===")
    for name, param in model.named_parameters():
        print(name)
    print("=== End of list ===")

if __name__ == "__main__":
    main()
