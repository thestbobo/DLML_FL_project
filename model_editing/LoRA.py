import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a LoRA low-rank adapter.
    """

    def __init__(self,
                 original_linear: nn.Linear,
                 r: int = 4,
                 alpha: float = 1.0,
                 dropout: float = 0.0):
        """
        Args:
            original_linear: the nn.Linear to wrap (its weights/bias are reused)
            r: rank of the adapter
            alpha: scaling factor (LoRA paper uses alpha/r scaling)
            dropout: dropout on adapter input
        """
        super().__init__()
        # save original
        self.in_features  = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r            = r
        self.alpha        = alpha
        self.scaling      = alpha / r

        # reuse original weight & bias (frozen by default)
        self.weight = original_linear.weight
        self.bias   = original_linear.bias

        # freeze original parameters
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA adapters
        self.lora_A = nn.Parameter(torch.zeros((r, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r)))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original linear output
        result = F.linear(x, self.weight, self.bias)
        # LoRA update: (x @ A^T @ B^T) * (alpha/r)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return result + lora_out * self.scaling

def replace_linear_with_lora(
    module: nn.Module,
    r: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_module_substrings: list[str] = None
):
    """
    Recursively replaces all nn.Linear layers whose name contains any of
    `target_module_substrings` (or all if None) with LoRALinear.
    """
    for name, child in module.named_children():
        # decide whether to replace
        is_linear = isinstance(child, nn.Linear)
        match_target = (
            target_module_substrings is None or
            any(sub in name for sub in target_module_substrings)
        )
        if is_linear and match_target:
            lora = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, lora)
        else:
            replace_linear_with_lora(
                child, r=r, alpha=alpha, dropout=dropout,
                target_module_substrings=target_module_substrings
            )

def mark_only_lora_as_trainable(model: nn.Module):
    """
    Freeze all parameters except LoRA adapters (lora_A and lora_B).
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def merge_lora_weights(model: nn.Module):
    """
    Merge LoRA adapters into the base weights for inference.
    After merging, the adapters are zeroed out.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # W <- W + (B @ A) * scaling
            delta = module.lora_B @ module.lora_A
            module.weight.data += delta * module.scaling
            # zero out adapters so they don't double-count
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()

def unmerge_lora_weights(model: nn.Module):
    """
    Attempt to rollback merge. NOTE: only works if you saved original weights
    or if merge was done in a reversible manner.
    Use with caution.
    """
    raise NotImplementedError(
        "Unmerge is not generally safe unless you kept a backup of original weights."
    )
