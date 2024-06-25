import torch

def tensor_statistics(tensor: torch.Tensor):
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item()
    }