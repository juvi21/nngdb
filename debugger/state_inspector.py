import torch

class StateInspector:
    def __init__(self, wrapped_model):
        self.wrapped_model = wrapped_model

    def inspect_neuron(self, layer_name: str, neuron_index: int = None):
        if layer_name in self.wrapped_model.current_state:
            layer_state = self.wrapped_model.current_state[layer_name]
            if neuron_index is None:
                return layer_state['output']
            else:
                return layer_state['output'][..., neuron_index]
        else:
            return None

    def inspect_gradient(self, layer_name: str, neuron_index: int = None):
        if layer_name in self.wrapped_model.current_state:
            layer_state = self.wrapped_model.current_state[layer_name]
            if 'grad_output' in layer_state:
                if neuron_index is None:
                    return layer_state['grad_output'][0]
                else:
                    return layer_state['grad_output'][0][..., neuron_index]
        return None

    def inspect_attention(self, layer_name: str):
        if layer_name in self.wrapped_model.current_state:
            layer_state = self.wrapped_model.current_state[layer_name]
            if isinstance(layer_state['output'], tuple) and len(layer_state['output']) > 1:
                attention_output = layer_state['output'][1]
                if isinstance(attention_output, torch.Tensor):
                    return attention_output
                elif hasattr(attention_output, 'to_tensor'):
                    return attention_output.to_tensor()
        return None

    def get_layer_parameters(self, layer_name: str):
        if layer_name in self.wrapped_model.current_state:
            return self.wrapped_model.current_state[layer_name]['parameters']
        return None