import torch

def analyze_activation(activation: torch.Tensor):
    return {
        "shape": activation.shape,
        "mean": activation.mean().item(),
        "std": activation.std().item(),
        "min": activation.min().item(),
        "max": activation.max().item(),
        "fraction_zeros": (activation == 0).float().mean().item(),
    }