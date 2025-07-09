import os
import torch

class PrintLogger:
    """Logs to the console."""
    def log_client_metrics(self, cid, metrics, t_round):
        print(f"Client {cid} Round {t_round}: {metrics}")

    def log_global_metrics(self, metrics, t_round):
        print(f"[Global] Round {t_round}: {metrics}")

class CheckpointManager:
    """Handles saving model checkpoints."""
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, t_round):
        path = os.path.join(self.checkpoint_dir, f"fl_model_round_{t_round}.pth")
        torch.save(model.state_dict(), path)
        print(f"Saved checkpoint: {path}")