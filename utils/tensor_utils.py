import torch

def tensor_stats(tensor: torch.Tensor):
    return {
        "shape": tensor.shape,
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "norm": tensor.norm().item(),
    }

def tensor_histogram(tensor: torch.Tensor, num_bins: int = 10):
    hist = torch.histogram(tensor.float().view(-1), bins=num_bins)
    return {
        "bin_edges": hist.bin_edges.tolist(),
        "counts": hist.count.tolist()
    }