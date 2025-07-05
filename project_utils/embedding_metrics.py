import wandb
import torch
import torch.nn.functional as F

def log_alignment_losses(log_dict, step=None):
    """
    Log individual alignment losses (G/D, rec, cycle, VSP) to Weights & Biases.

    Args:
        log_dict (dict): A dictionary of losses (e.g., {'g_loss': ..., 'rec_loss': ...}).
        step (int, optional): Training step or epoch.
    """
    if step is not None:
        log_dict['alignment_step'] = step
    wandb.log(log_dict)


def log_cosine_similarity(mean_cos_sim, tag='alignment/test', step=None):
    """
    Log cosine similarity between aligned embeddings and real targets.

    Args:
        mean_cos_sim (float): Average cosine similarity.
        tag (str): Metric label (e.g. 'alignment/test' or 'alignment/train').
        step (int, optional): Step/epoch number.
    """
    log_data = {tag: mean_cos_sim}
    if step is not None:
        log_data['alignment_step'] = step
    wandb.log(log_data)


def adversarial_loss(preds, is_real):
    '''
    Standard binary cross-entropy loss for GANs.
    '''
    targets = torch.ones_like(preds) if is_real else torch.zeros_like(preds)
    return F.binary_cross_entropy(preds, targets)

def reconstruction_loss(x, x_reconstructed):
    '''
    Ensures translated embedding can reconstruct the original.
    '''
    return F.mse_loss(x, x_reconstructed)

def cycle_consistency_loss(original, cycled):
    '''
    Ensures that translating to another space and back recovers the original.
    '''
    return F.mse_loss(original, cycled)

def vector_space_preservation(x, x_translated):
    '''
    Preserves relative structure between embeddings via dot-product similarity.
    '''
    dot_orig = torch.mm(x, x.t())
    dot_trans = torch.mm(x_translated, x_translated.t())
    return F.mse_loss(dot_orig, dot_trans)
