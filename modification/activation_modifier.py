import torch
from core.model_wrapper import ModelWrapper

class ActivationModifier:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.hooks = {}

    def modify_activation(self, layer_name: str, modification_function):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        def hook(module, input, output):
            return modification_function(output)

        handle = layer.register_forward_hook(hook)
        self.hooks[layer_name] = handle

        return f"Activation modification hook set for layer '{layer_name}'"

    def remove_modification(self, layer_name: str):
        if layer_name in self.hooks:
            self.hooks[layer_name].remove()
            del self.hooks[layer_name]
            return f"Activation modification removed for layer '{layer_name}'"
        else:
            return f"No activation modification found for layer '{layer_name}'"

    def clear_all_modifications(self):
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
        return "All activation modifications cleared"

    def add_noise_to_activation(self, layer_name: str, noise_scale: float):
        def add_noise(output):
            return output + torch.randn_like(output) * noise_scale

        return self.modify_activation(layer_name, add_noise)

    def clip_activation(self, layer_name: str, min_val: float, max_val: float):
        def clip(output):
            return torch.clamp(output, min_val, max_val)

        return self.modify_activation(layer_name, clip)

    def scale_activation(self, layer_name: str, scale_factor: float):
        def scale(output):
            return output * scale_factor

        return self.modify_activation(layer_name, scale)