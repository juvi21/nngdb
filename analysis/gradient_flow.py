import torch
import matplotlib.pyplot as plt

def analyze_gradient_flow(model: torch.nn.Module):
    if not any(p.grad is not None for p in model.parameters()):
        return "No gradients available. Run a backward pass first."

    avg_grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            avg_grads.append(p.grad.abs().mean().item())
    
    plt.bar(range(len(avg_grads)), avg_grads, align='center')
    plt.xticks(range(len(avg_grads)), layers, rotation='vertical')
    plt.xlabel('Layers')
    plt.ylabel('Average gradient')
    plt.title('Gradient flow')
    plt.tight_layout()
    plt.show()

def detect_vanishing_exploding_gradients(model: torch.nn.Module, threshold=1e-4):
    vanishing = []
    exploding = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                if grad_norm < threshold:
                    vanishing.append((n, grad_norm))
                elif grad_norm > 1/threshold:
                    exploding.append((n, grad_norm))
    return vanishing, exploding