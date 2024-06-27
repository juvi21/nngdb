import torch
from core.model_wrapper import ModelWrapper

class ActivationInspector:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def inspect(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No activation data available for layer '{layer_name}'."

        activation = self.wrapped_model.current_state[layer_name]['output']
        if not isinstance(activation, torch.Tensor):
            return f"Activation for layer '{layer_name}' is not a tensor."

        return self._analyze_activation(activation)

    def _analyze_activation(self, activation: torch.Tensor):
        return {
            "shape": activation.shape,
            "mean": activation.mean().item(),
            "std": activation.std().item(),
            "min": activation.min().item(),
            "max": activation.max().item(),
            "num_zeros": (activation == 0).sum().item(),
            "num_non_zeros": (activation != 0).sum().item(),
            "fraction_zeros": ((activation == 0).sum() / activation.numel()).item(),
        }

    def get_activation(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No activation data available for layer '{layer_name}'."
        return self.wrapped_model.current_state[layer_name]['output']