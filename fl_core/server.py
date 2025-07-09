import torch
from collections import OrderedDict
from typing import List, Dict, Any, Optional

class Server:
    """
    Federated learning server. Aggregates model updates using FedAvg.
    """
    @staticmethod
    def aggregate(updates: List[Dict[str, torch.Tensor]], weights: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Performs (weighted) FedAvg aggregation.
        Args:
            updates: List of state dicts from clients.
            weights: List of sample counts per client (for weighted avg). If None, uses simple average.
        Returns:
            Aggregated state dict.
        """
        if not updates:
            raise ValueError("No client updates provided.")
        avg = OrderedDict()
        keys = updates[0].keys()
        if weights is None:
            # Simple average
            for k in keys:
                avg[k] = torch.stack([u[k] for u in updates], dim=0).mean(dim=0)
        else:
            total = sum(weights)
            for k in keys:
                avg[k] = sum(u[k] * (w / total) for u, w in zip(updates, weights))
        return avg