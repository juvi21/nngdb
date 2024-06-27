import torch
from core.model_wrapper import ModelWrapper

class WeightModifier:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def modify_weight(self, layer_name: str, weight_name: str, indices, value):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, weight_name):
            return f"Weight '{weight_name}' not found in layer '{layer_name}'."

        weight = getattr(layer, weight_name)
        if not isinstance(weight, torch.Tensor):
            return f"'{weight_name}' is not a tensor in layer '{layer_name}'."

        try:
            with torch.no_grad():
                weight[indices] = value
            return f"Weight at {layer_name}.{weight_name}{indices} modified to {value}"
        except Exception as e:
            return f"Error modifying weight: {str(e)}"

    def scale_weights(self, layer_name: str, weight_name: str, scale_factor: float):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, weight_name):
            return f"Weight '{weight_name}' not found in layer '{layer_name}'."

        weight = getattr(layer, weight_name)
        if not isinstance(weight, torch.Tensor):
            return f"'{weight_name}' is not a tensor in layer '{layer_name}'."

        with torch.no_grad():
            weight.mul_(scale_factor)

        return f"Weights in {layer_name}.{weight_name} scaled by {scale_factor}"

    def reset_weights(self, layer_name: str):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        def weight_reset(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        layer.apply(weight_reset)
        return f"Weights in layer '{layer_name}' have been reset."

    def add_noise_to_weights(self, layer_name: str, weight_name: str, noise_scale: float):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, weight_name):
            return f"Weight '{weight_name}' not found in layer '{layer_name}'."

        weight = getattr(layer, weight_name)
        if not isinstance(weight, torch.Tensor):
            return f"'{weight_name}' is not a tensor in layer '{layer_name}'."

        with torch.no_grad():
            noise = torch.randn_like(weight) * noise_scale
            weight.add_(noise)

        return f"Noise added to weights in {layer_name}.{weight_name} with scale {noise_scale}"

    def prune_weights(self, layer_name: str, weight_name: str, threshold: float):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, weight_name):
            return f"Weight '{weight_name}' not found in layer '{layer_name}'."

        weight = getattr(layer, weight_name)
        if not isinstance(weight, torch.Tensor):
            return f"'{weight_name}' is not a tensor in layer '{layer_name}'."

        with torch.no_grad():
            mask = (weight.abs() > threshold).float()
            weight.mul_(mask)

        pruned_percentage = (1 - mask.mean().item()) * 100
        return f"Pruned {pruned_percentage:.2f}% of weights in {layer_name}.{weight_name}"