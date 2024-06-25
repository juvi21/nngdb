import torch
import matplotlib.pyplot as plt

def get_top_k_activated_neurons(activation: torch.Tensor, k: int = 10):
    flattened = activation.view(-1)
    top_k_values, top_k_indices = torch.topk(flattened, k)
    return top_k_values, top_k_indices

def visualize_neuron_activations(activations, layer_name: str):
    if isinstance(activations, tuple):
        activations = activations[0]  # Assume the first element is the main output
    if not isinstance(activations, torch.Tensor):
        return f"Unable to visualize activations for {layer_name}: unexpected type {type(activations)}"
    
    activations = activations.squeeze().cpu().detach().numpy()
    
    plt.figure(figsize=(12, 6))
    if len(activations.shape) == 2:
        plt.imshow(activations, cmap='viridis', aspect='auto')
    elif len(activations.shape) == 1:
        plt.plot(activations)
    else:
        return f"Unable to visualize activations for {layer_name}: unexpected shape {activations.shape}"
    
    plt.colorbar()
    plt.title(f'Neuron Activations in {layer_name}')
    plt.xlabel('Neuron Index')
    plt.ylabel('Sequence Position' if len(activations.shape) == 2 else 'Activation Value')
    plt.tight_layout()
    plt.show()