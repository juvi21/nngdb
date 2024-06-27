import torch
from core.model_wrapper import ModelWrapper

class GradientInspector:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def inspect(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No gradient data available for layer '{layer_name}'."

        grad_info = {}
        if 'grad_input' in self.wrapped_model.current_state[layer_name]:
            grad_input = self.wrapped_model.current_state[layer_name]['grad_input']
            grad_info['input_gradient'] = self._analyze_gradient(grad_input)

        if 'grad_output' in self.wrapped_model.current_state[layer_name]:
            grad_output = self.wrapped_model.current_state[layer_name]['grad_output']
            grad_info['output_gradient'] = self._analyze_gradient(grad_output)

        if 'grad_params' in self.wrapped_model.current_state[layer_name]:
            grad_params = self.wrapped_model.current_state[layer_name]['grad_params']
            grad_info['parameter_gradients'] = {name: self._analyze_gradient(grad) for name, grad in grad_params.items()}

        return grad_info

    def _analyze_gradient(self, gradient):
        if not isinstance(gradient, torch.Tensor):
            return "Gradient is not a tensor."

        return {
            "shape": gradient.shape,
            "mean": gradient.mean().item(),
            "std": gradient.std().item(),
            "min": gradient.min().item(),
            "max": gradient.max().item(),
            "norm": gradient.norm().item(),
            "num_zeros": (gradient == 0).sum().item(),
            "num_non_zeros": (gradient != 0).sum().item(),
        }

    def get_gradient(self, layer_name: str, grad_type: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No gradient data available for layer '{layer_name}'."
        
        if grad_type == 'input':
            return self.wrapped_model.current_state[layer_name].get('grad_input')
        elif grad_type == 'output':
            return self.wrapped_model.current_state[layer_name].get('grad_output')
        elif grad_type == 'params':
            return self.wrapped_model.current_state[layer_name].get('grad_params')
        else:
            return f"Invalid gradient type '{grad_type}'. Choose from 'input', 'output', or 'params'."