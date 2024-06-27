import torch
from core.model_wrapper import ModelWrapper

class LayerInspector:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def inspect(self, layer_name: str):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        layer_info = {
            "name": layer_name,
            "type": type(layer).__name__,
            "parameters": self._get_parameters_info(layer),
            "input_shape": self._get_input_shape(layer_name),
            "output_shape": self._get_output_shape(layer_name)
        }
        return layer_info

    def _get_parameters_info(self, layer):
        return {name: {"shape": param.shape, "requires_grad": param.requires_grad}
                for name, param in layer.named_parameters()}

    def _get_input_shape(self, layer_name):
        if layer_name in self.wrapped_model.current_state:
            inputs = self.wrapped_model.current_state[layer_name]['input']
            return [tuple(input.shape) for input in inputs]
        return None

    def _get_output_shape(self, layer_name):
        if layer_name in self.wrapped_model.current_state:
            output = self.wrapped_model.current_state[layer_name]['output']
            return tuple(output.shape) if isinstance(output, torch.Tensor) else None
        return None