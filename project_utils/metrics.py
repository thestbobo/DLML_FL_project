from sklearn.metrics import f1_score

def get_metrics(outputs, labels, top_k=(1, 5)):
    """
    Calcola top-1 accuracy, top-5 accuracy e F1 score.

    Args:
        outputs (torch.Tensor): Output del modello (logits o probabilità) di forma (batch_size, num_classes).
        labels (torch.Tensor): Etichette vere di forma (batch_size,).
        top_k (tuple): Tuple contenente i valori di k per calcolare le top-k accuracy.

    Returns:
        dict: Dizionario con top-1 accuracy, top-5 accuracy e F1 score.
    """
    # Calcolo delle top-k accuracy
    max_k = max(top_k)
    _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()  # Trasponi per confrontare con le etichette
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    top_k_accuracies = {}
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        top_k_accuracies[f"top_{k}_accuracy"] = (correct_k / labels.size(0)).item()

    # Calcolo dell'F1 score
    _, top1_pred = outputs.max(dim=1)
    f1 = f1_score(labels.cpu().numpy(), top1_pred.cpu().numpy(), average="weighted")

    # Combina i risultati
    metrics = {
        **top_k_accuracies,
        "f1_score": f1
    }
    return metrics
