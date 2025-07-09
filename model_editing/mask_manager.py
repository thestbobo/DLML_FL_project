import os
import torch

class MaskManager:
    """
    Handles saving, loading, and computation of global (or local) masks for pruning/sparsity.
    Used for federated and local sparse training.
    """
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        self.mask_dir = config.get('MASKS_DIR', 'masks/')
        os.makedirs(self.mask_dir, exist_ok=True)

    def get_or_create_global_mask(self, client_datasets, mask_generator=None):
        """
        Loads a precomputed global mask if it exists, otherwise computes and saves a new mask.

        Args:
            client_datasets: list of datasets for clients (optional, for mask computation).
            mask_generator: function(model, client_datasets, device) -> mask dict, optional custom mask generator.

        Returns:
            dict: name (str) -> torch.Tensor (mask)
        """
        mask_file = os.path.join(self.mask_dir, "mask_global.pt")
        if os.path.exists(mask_file):
            return torch.load(mask_file, map_location=self.device)
        if mask_generator is not None:
            mask = mask_generator(self.model, client_datasets, self.device)
        else:
            # Default: all ones mask (no pruning)
            mask = {name: torch.ones_like(param, dtype=torch.float32)
                    for name, param in self.model.named_parameters()}
        torch.save(mask, mask_file)
        return mask

    def save_mask(self, mask, mask_name="mask.pt"):
        mask_file = os.path.join(self.mask_dir, mask_name)
        torch.save(mask, mask_file)

    def load_mask(self, mask_name="mask.pt"):
        mask_file = os.path.join(self.mask_dir, mask_name)
        return torch.load(mask_file, map_location=self.device)