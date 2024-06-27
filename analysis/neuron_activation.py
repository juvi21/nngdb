import torch
from core.model_wrapper import ModelWrapper

class NeuronActivationAnalyzer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def analyze(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No activation data available for layer '{layer_name}'."

        activation = self.wrapped_model.current_state[layer_name]['output']
        if not isinstance(activation, torch.Tensor):
            return f"Activation for layer '{layer_name}' is not a tensor."

        return self._analyze_activation(activation)

    def _analyze_activation(self, activation: torch.Tensor):
        analysis = {
            "shape": activation.shape,
            "mean": activation.mean().item(),
            "std": activation.std().item(),
            "min": activation.min().item(),
            "max": activation.max().item(),
            "fraction_zeros": (activation == 0).float().mean().item(),
            "top_k_active": self._get_top_k_active(activation, k=10),
            "activation_statistics": self._compute_activation_statistics(activation),
        }
        return analysis

    def _get_top_k_active(self, activation: torch.Tensor, k: int):
        if activation.dim() > 2:
            activation = activation.view(activation.size(0), -1)
        top_k_values, top_k_indices = torch.topk(activation.abs().mean(dim=0), k)
        return [(idx.item(), val.item()) for idx, val in zip(top_k_indices, top_k_values)]

    def _compute_activation_statistics(self, activation: torch.Tensor):
        if activation.dim() > 2:
            activation = activation.view(activation.size(0), -1)
        
        positive_fraction = (activation > 0).float().mean(dim=0)
        negative_fraction = (activation < 0).float().mean(dim=0)
        zero_fraction = (activation == 0).float().mean(dim=0)

        return {
            "positive_fraction": positive_fraction.mean().item(),
            "negative_fraction": negative_fraction.mean().item(),
            "zero_fraction": zero_fraction.mean().item(),
        }

    def get_most_active_neurons(self, layer_name: str, k: int = 10):
        if layer_name not in self.wrapped_model.current_state:
            return f"No activation data available for layer '{layer_name}'."

        activation = self.wrapped_model.current_state[layer_name]['output']
        if not isinstance(activation, torch.Tensor):
            return f"Activation for layer '{layer_name}' is not a tensor."

        if activation.dim() > 2:
            activation = activation.view(activation.size(0), -1)

        mean_activation = activation.abs().mean(dim=0)
        top_k_values, top_k_indices = torch.topk(mean_activation, k)

        return [(idx.item(), val.item()) for idx, val in zip(top_k_indices, top_k_values)]

    def compute_activation_similarity(self, layer_name: str, reference_input: torch.Tensor):
        if layer_name not in self.wrapped_model.current_state:
            return f"No activation data available for layer '{layer_name}'."

        current_activation = self.wrapped_model.current_state[layer_name]['output']
        if not isinstance(current_activation, torch.Tensor):
            return f"Activation for layer '{layer_name}' is not a tensor."

        with torch.no_grad():
            reference_activation = self.wrapped_model.model(reference_input)
            reference_activation = self.wrapped_model.current_state[layer_name]['output']

        similarity = torch.nn.functional.cosine_similarity(current_activation.view(-1), reference_activation.view(-1), dim=0)

        return similarity.item()