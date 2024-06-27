import torch
import torch.nn as nn
from core.model_wrapper import ModelWrapper

class ModelSurgeon:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def replace_layer(self, layer_name: str, new_layer: nn.Module):
        parent_name, child_name = layer_name.rsplit('.', 1)
        parent_module = self.wrapped_model.get_layer(parent_name)

        if parent_module is None:
            return f"Parent module of '{layer_name}' not found."

        if not hasattr(parent_module, child_name):
            return f"Layer '{child_name}' not found in '{parent_name}'."

        setattr(parent_module, child_name, new_layer)
        return f"Layer '{layer_name}' replaced with {type(new_layer).__name__}"

    def insert_layer(self, layer_name: str, new_layer: nn.Module, position: str = 'after'):
        parent_name, child_name = layer_name.rsplit('.', 1)
        parent_module = self.wrapped_model.get_layer(parent_name)

        if parent_module is None:
            return f"Parent module of '{layer_name}' not found."

        if not hasattr(parent_module, child_name):
            return f"Layer '{child_name}' not found in '{parent_name}'."

        original_layer = getattr(parent_module, child_name)
        
        class WrappedLayer(nn.Module):
            def __init__(self, original_layer, new_layer, position):
                super().__init__()
                self.original_layer = original_layer
                self.new_layer = new_layer
                self.position = position

            def forward(self, x):
                if self.position == 'before':
                    x = self.new_layer(x)
                    return self.original_layer(x)
                elif self.position == 'after':
                    x = self.original_layer(x)
                    return self.new_layer(x)

        wrapped_layer = WrappedLayer(original_layer, new_layer, position)
        setattr(parent_module, child_name, wrapped_layer)

        return f"Layer '{new_layer.__class__.__name__}' inserted {position} '{layer_name}'"

    def remove_layer(self, layer_name: str):
        parent_name, child_name = layer_name.rsplit('.', 1)
        parent_module = self.wrapped_model.get_layer(parent_name)

        if parent_module is None:
            return f"Parent module of '{layer_name}' not found."

        if not hasattr(parent_module, child_name):
            return f"Layer '{child_name}' not found in '{parent_name}'."

        class Identity(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        setattr(parent_module, child_name, Identity())
        return f"Layer '{layer_name}' removed and replaced with Identity"

    def change_activation_function(self, layer_name: str, new_activation: nn.Module):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, 'activation'):
            return f"Layer '{layer_name}' does not have an 'activation' attribute."

        layer.activation = new_activation
        return f"Activation function of '{layer_name}' changed to {type(new_activation).__name__}"