
import yaml
import os
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


import torch
from embeddings.alignment_models.adapters import Adapter
from embeddings.alignment_models.backbone import Backbone

def translate_task_vector(model_name, vector_path):
    save_path = cfg["paths"]["aligned_vectors_dir"]
    checkpoint_path = os.path.join(cfg["paths"]["checkpoints_dir"], "alignment", "vec2vec_epoch_100.pt")

    os.makedirs(save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load task vector
    task_vector = torch.load(vector_path).to(device)

    # Load alignment modules
    A = Adapter(input_dim=384, latent_dim=512).to(device)
    T = Backbone(latent_dim=512).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    A.load_state_dict(ckpt[f"A1"] if model_name == 'dino' else ckpt[f"A2"])  # A1 for dino, A2 for deit
    T.load_state_dict(ckpt["T"])

    A.eval()
    T.eval()

    with torch.no_grad():
        latent_vector = T(A(task_vector))

    aligned_path = os.path.join(save_path, f"{model_name}_task_vector_aligned.pt")
    torch.save(latent_vector.cpu(), aligned_path)
    print(f"[✓] Saved aligned task vector to {aligned_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model name (dino, deit, mae)")
    parser.add_argument('--vector_path', type=str, required=True, help="Path to task vector .pt file")
    args = parser.parse_args()

    translate_task_vector(args.model, args.vector_path)

