import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from model_editing.SparseSGDM import SparseSGDM

class Client:
    """
    Federated learning client. Performs local training and returns updated weights.
    """
    def __init__(
        self,
        cid: int,
        model: torch.nn.Module,
        train_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        mask: Optional[Dict[str, torch.Tensor]] = None,
        optimizer_class = SparseSGDM,
        scheduler_class = None
    ):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.config = config
        self.mask = mask

        lr = config['FEDERATED_TRAINING']['LR']
        momentum = config.get('talos_fine_tuning', {}).get('nesterov', 0)
        weight_decay = config['FEDERATED_TRAINING'].get('LR_DECAY', 0)

        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            mask=self.mask,
            model=self.model
        )
        # Scheduler is optional
        self.scheduler = None
        if scheduler_class is not None:
            self.scheduler = scheduler_class(self.optimizer, config)

    def local_update(self, global_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Loads global model params, performs local training, returns updated state dict.
        """
        self.model.load_state_dict(global_params)
        self.model.to(self.device)
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        local_steps = self.config['FEDERATED_TRAINING']['LOCAL_STEPS']

        for _ in range(local_steps):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
        self.print_optimizer_stats()
        return self.model.state_dict()
    
    def print_optimizer_stats(self):
        print(f"[Client {self.cid}] Optimizer parameter stats:")
        for i, group in enumerate(self.optimizer.param_groups):
            for param in group['params']:
                grad_norm = None
                if param.grad is not None:
                    grad_norm = param.grad.abs().mean().item()
                mask_info = ""
                pname = getattr(param, 'name', None)
                if self.mask and pname and pname in self.mask:
                    mask_info = f" (mask kept: {int(self.mask[pname].sum())}/{self.mask[pname].numel()})"
                print(f"  shape: {list(param.shape)}, requires_grad: {param.requires_grad}, grad_mean: {grad_norm}{mask_info}")