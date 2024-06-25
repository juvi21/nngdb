import torch
import matplotlib.pyplot as plt
from typing import Dict
import seaborn as sns

def visualize_weights(weights: Dict[str, torch.Tensor]):
    for name, weight in weights.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(weight.detach().cpu().numpy(), cmap="viridis", center=0)
        plt.title(f"Weights: {name}")
        plt.show()

def visualize_activations(activations: torch.Tensor, layer_name: str):
    plt.figure(figsize=(12, 6))
    sns.heatmap(activations.detach().cpu().numpy(), cmap="viridis")
    plt.title(f"Activations: {layer_name}")
    plt.xlabel("Neuron")
    plt.ylabel("Sample")
    plt.show()