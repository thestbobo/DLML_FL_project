import os
import torch
from model_editing.TaLoS import compute_fisher_scores, calibrate_mask_global

class MaskManager:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        self.mask_dir = config.get('MASKS_DIR', 'masks/')
        os.makedirs(self.mask_dir, exist_ok=True)

    def get_or_create_global_mask(self, dataloader, criterion):
        mask_file = os.path.join(self.mask_dir, "mask_global.pt")
        if os.path.exists(mask_file):
            return torch.load(mask_file, map_location=self.device)

        fisher_scores = compute_fisher_scores(self.model, dataloader, criterion, self.device)
        mask = calibrate_mask_global(
            fisher_scores,
            target_sparsity=self.config.TALOS_TARGET_SPARSITY
        )
        torch.save(mask, mask_file)
        return mask

    def save_mask(self, mask, mask_name="mask.pt"):
        mask_file = os.path.join(self.mask_dir, mask_name)
        torch.save(mask, mask_file)

    def load_mask(self, mask_name="mask.pt"):
        mask_file = os.path.join(self.mask_dir, mask_name)
        return torch.load(mask_file, map_location=self.device)