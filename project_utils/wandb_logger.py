import wandb

class WandBLogger:
    def __init__(self, project, run_id=None, config=None):
        if run_id:
            wandb.init(project=project, id=run_id, config=config)
        else:
            wandb.init(project=project, config=config)

    def log_local_weight_norms(self, local_weights, round_num):
        norms = []
        for weights in local_weights:
            total_sq = sum((param.norm().item() ** 2 for param in weights.values()))
            norms.append(total_sq ** 0.5)
        wandb.log({'round': round_num, 'local_weight_norms': norms})

    def log_global_weight_diff(self, prev_weights, new_weights, round_num):
        total_sq = 0.0
        for key in prev_weights.keys():
            diff = prev_weights[key].cpu() - new_weights[key].cpu()
            total_sq += diff.norm().item() ** 2
        diff_norm = total_sq ** 0.5
        wandb.log({'round': round_num, 'global_weight_diff': diff_norm})

    def log_aggregated_class_distribution(self, client_datasets, selected_clients, round_num):
        all_labels = []
        for cid in selected_clients:
            all_labels.extend([label for _, label in client_datasets[cid]])
        wandb.log({'round': round_num, 'aggregated_class_distribution': wandb.Histogram(all_labels)})

    def log_round_info(self, round_num, num_clients, num_samples):
        wandb.log({'round': round_num, 'clients_participated': num_clients, 'samples_used': num_samples})

    def log_client_metrics(self, client_id, loss, accuracy, round_num):
        wandb.log({'round': round_num,
                   f'client_{client_id}/local_loss': loss,
                   f'client_{client_id}/local_accuracy': accuracy})

    def log_global_metrics(self, metrics, round_num):
        data = {'round': round_num}
        for key, value in metrics.items():
            data[f'global_test_{key}'] = value
        wandb.log(data)