import os
import torch


def get_intermediate_representation(model, x, layer_names, device):
    """
    Return a dict of {layer_name: representation} for given input batch x.
    """
    model.eval().to(device)
    outputs = {}
    handles = []

    def get_hook(name):
        def hook(module, input, output):
            outputs[name] = output.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(get_hook(name)))

    with torch.no_grad():
        _ = model(x.to(device))

    for h in handles:
        h.remove()

    return outputs


def save_representations(reps, save_path, client_id, round_num, class_counts=None):
    os.makedirs(save_path, exist_ok=True)
    save_obj = {
        "representations": reps,
        "client_id": client_id,
        "round": round_num
    }
    if class_counts:
        save_obj["class_counts"] = class_counts

    filename = os.path.join(save_path, f"client{client_id}_round{round_num}.pt")
    torch.save(save_obj, filename)
    print(f"[INFO] Saved representations for Client {client_id}, Round {round_num} at {filename}")

