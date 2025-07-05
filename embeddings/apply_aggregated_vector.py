import yaml
import os
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


import torch
from embeddings.alignment_models.adapters import Adapter
from embeddings.alignment_models.backbone import Backbone

def apply_aggregated_vector(model_name, base_embedding_path):
    agg_vector_path = cfg["paths"]["aggregated_vector_path"]
    checkpoint_path = os.path.join(cfg["paths"]["checkpoints_dir"], "alignment", "vec2vec_epoch_100.pt")
    output_path = os.path.join(cfg["paths"]["embeddings_dir"], f"{model_name}_modified.pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load base embeddings
    base = torch.load(base_embedding_path).to(device)

    # Load aligned task vector
    delta = torch.load(agg_vector_path).to(device)

    # Load model adapters
    B = Adapter(latent_dim=512, input_dim=384).to(device)  # decoder
    T = Backbone(latent_dim=512).to(device)

    # Load correct checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    B.load_state_dict(ckpt[f'B1'] if model_name == 'dino' else ckpt[f'B2'])  # use corresponding decoder
    T.load_state_dict(ckpt['T'])

    B.eval()
    T.eval()

    with torch.no_grad():
        base_to_latent = T(torch.zeros_like(delta))  # identity base placeholder
        modified_latent = base_to_latent + delta
        edited_embedding = B(modified_latent)

    torch.save(edited_embedding.cpu(), output_path)
    print(f"[✓] Saved edited embedding for {model_name} to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Target model (dino, deit, mae)")
    parser.add_argument('--base_embedding', type=str, required=True, help="Path to base embeddings (before tuning)")
    args = parser.parse_args()

    apply_aggregated_vector(args.model, args.base_embedding)

