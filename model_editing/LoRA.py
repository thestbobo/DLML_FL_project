# LoRA.py

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LoraConfig:
    r: int
    alpha: int
    target_modules: List[str]        # substrings of module names to wrap
    dropout: float = 0.0             # dropout on the LoRA path


class LoRALayer(nn.Module):
    """
    Wraps an nn.Linear with a low-rank update W' = W + (A @ B) * (alpha / r).
    Only A and B are trainable.
    """
    def __init__(self,
                 orig_linear: nn.Linear,
                 r: int,
                 alpha: int,
                 dropout: float = 0.0):
        super().__init__()
        self.orig = orig_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA parameters
        self.A = nn.Parameter(torch.zeros(orig_linear.in_features, r))
        self.B = nn.Parameter(torch.zeros(r, orig_linear.out_features))
        # init
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # frozen original + low-rank path
        return self.orig(x) + self.dropout(x @ self.A @ self.B) * self.scaling


def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    """Traverse root by dot-separated path to return the submodule."""
    for attr in path.split('.'):
        root = getattr(root, attr)
    return root


def _set_submodule(root: nn.Module, path: str, new_mod: nn.Module):
    """Replace the submodule at `path` under `root` with `new_mod`."""
    parts = path.split('.')
    parent = _get_submodule(root, '.'.join(parts[:-1])) if len(parts) > 1 else root
    setattr(parent, parts[-1], new_mod)


def apply_lora(model: nn.Module, cfg: LoraConfig) -> nn.Module:
    """
    Scan `model` for all nn.Linear whose full name contains any of cfg.target_modules,
    replace them with LoRALayer(orig_linear, cfg.r, cfg.alpha, cfg.dropout).
    Returns the wrapped model.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in cfg.target_modules):
            lora_mod = LoRALayer(module, cfg.r, cfg.alpha, cfg.dropout)
            _set_submodule(model, name, lora_mod)
    return model


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Collect and return all LoRA parameters (A & B) in `model`.
    """
    params = []
    for m in model.modules():
        if isinstance(m, LoRALayer):
            params += [m.A, m.B]
    return params

