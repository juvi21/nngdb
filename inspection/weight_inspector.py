import torch
from core.model_wrapper import ModelWrapper

class WeightInspector:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def inspect(self, layer_name: str):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        weight_info = {}
        for name, param in layer.named_parameters():
            if 'weight' in name:
                weight_info[name] = self._analyze_weight(param)
        return weight_info

    def _analyze_weight(self, weight: torch.Tensor):
        return {
            "shape": weight.shape,
            "mean": weight.mean().item(),
            "std": weight.std().item(),
            "min": weight.min().item(),
            "max": weight.max().item(),
            "norm": weight.norm().item(),
            "num_zeros": (weight == 0).sum().item(),
            "num_non_zeros": (weight != 0).sum().item(),
        }

    def get_weight(self, layer_name: str, weight_name: str):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."
        
        for name, param in layer.named_parameters():
            if name == weight_name:
                return param
        return f"Weight '{weight_name}' not found in layer '{layer_name}'."