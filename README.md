# DLML_FL_project

This repository supports a research project on **Deep Learning and Federated Learning**, focusing on **centralized and federated training of Vision Transformers (ViT)** on the CIFAR-100 dataset. The project explores advanced fine-tuning and pruning methods such as **LoRA** and **TaLoS**, targeting efficient adaptation in distributed, resource-constrained scenarios.

---

## Project Structure

```
DLML_FL_project/
│
├── colab_runner.ipynb      # Jupyter notebook for Colab environment setup and running experiments
├── train_centralized.py    # Centralized (single-node) training script
├── train_federated.py      # Federated learning training script
├── test.py                 # Model evaluation and checkpoint testing
├── requirements.txt        # Python dependencies
├── config/                 # Configuration files (e.g., config.yaml)
├── data/                   # Dataset storage and preprocessing scripts
├── model_editing/          # Fine-tuning and pruning modules (e.g., LoRA, TaLoS)
├── models/                 # Model architectures (e.g., Vision Transformers)
├── fl_core/                # Federated learning core logic (client/server)
├── project_utils/          # Utility functions (metrics, schedulers, logging)
├── .idea/, .gitignore      # Development and VCS files
└── README.md
```

## Key Features

- **Hybrid Training Support**: Run both centralized and federated experiments with the same architecture.
- **Fine-Tuning Strategies**:
  - *LoRA*: Low-Rank Adaptation for efficient tuning.
  - *TaLoS*: Layer-wise pruning based on sensitivity.
  - *Dense*: Classic full-model fine-tuning.
- **Federated Learning Core**: Simulated client-server setup supporting custom aggregation and gradient masking.
- **ViT with DINO**: Leverages powerful self-supervised features from DINO-pretrained ViT backbones.
- **WandB Integration**: Seamless experiment logging and sweep tracking with [Weights & Biases](https://wandb.ai/).
- **Flexible Configuration**: YAML-based experiment configuration for easy reproducibility.
- **Colab Ready**: Plug-and-play support for Google Colab users.

---

## Usage

### Google Colab

A ready-made notebook (`colab_runner.ipynb`) is provided for experiments in Google Colab:
- Mounts Google Drive for data and checkpoint storage.
- Installs requirements and runs training scripts.
- Includes guidance for using wandb and running sweeps.

### Configuration

Edit `config/config.yaml` to adjust hyperparameters such as batch size, learning rate, number of epochs, fine-tuning strategy, data splits, checkpoint paths, federated learning parameters, etc.

Example:
```yaml
batch_size: 256
val_split: 0.2
num_workers: 4
learning_rate: 0.008805
weight_decay: 0.01
momentum: 0.9
epochs: 50
finetuning_method: 'lora' # or 'dense', 'talos'
checkpoint_path: './checkpoints'
...
```

## Resources

- Project Google Drive folder: [link](https://drive.google.com/drive/folders/1JXbfCxJPu4f1d09HdaeyZtpiLiD5xQoI?usp=share_link)
- Overleaf report: [link](https://www.overleaf.com/2723819825wzxfqbwzknhn#add6bc)

---

## References

1. Hsu et al., *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification*, 2019.  
2. Reddi et al., *Adaptive Federated Optimization*, ICLR 2021.  
3. Kairouz et al., *Advances and Open Problems in Federated Learning*, Foundations and Trends in Machine Learning, 2021.  
4. Li et al., *Federated Optimization in Heterogeneous Networks*, PMLR, 2018.  
5. Karimireddy et al., *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*, PMLR, 2020.  
6. Kairouz et al., *Advances and Open Problems in Federated Learning*, Foundations and Trends in Machine Learning, 2021.  
7. Li et al., *A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection*, 2019.  
8. Chen et al., *Privacy and Fairness in Federated Learning: On the Perspective of Tradeoff*, ACM Computing Surveys, 2023.  
9. Pfeiffer et al., *Federated Learning for Computationally Constrained Heterogeneous Devices: A Survey*, ACM Computing Surveys, 2023.  
10. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS, 2017.  
11. Hsu et al., *Federated Visual Classification with Real-World Data Distribution*, ECCV, 2020.  
12. Qu et al., *Rethinking Architecture Design for Tackling Data Heterogeneity in Federated Learning*, CVPR, 2022.  
13. Pieri et al., *Handling Data Heterogeneity via Architectural Design for Federated Visual Recognition*, NeurIPS, 2023.  
14. Li et al., *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization*, ICLR, 2021.  
15. Iurada et al., *Efficient Model Editing with Task-Localized Sparse Fine-tuning*, ICLR, 2025.  

