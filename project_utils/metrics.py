from sklearn.metrics import f1_score

def get_topk_accuracies(outputs, labels, top_k=(1, 5)):
    max_k = max(top_k)
    _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    top_k_accuracies = {}
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        top_k_accuracies[f"top_{k}_accuracy"] = (correct_k / labels.size(0)).item()
    return top_k_accuracies

def get_metrics(outputs, labels, top_k=(1, 5)):
    top_k_accuracies = get_topk_accuracies(outputs, labels, top_k)
    _, top1_pred = outputs.max(dim=1)
    f1 = f1_score(labels.cpu().numpy(), top1_pred.cpu().numpy(), average="weighted")
    metrics = {
        **top_k_accuracies,
        "f1_score": f1
    }
    return metrics
