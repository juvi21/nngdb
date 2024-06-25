import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import copy

class NNGDBWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.initial_state_dict = copy.deepcopy(model.state_dict())
        self.breakpoints: Dict[str, List[Dict[str, Any]]] = {}
        self.current_state: Dict[str, Any] = {}
        self.execution_paused = False
        self.step_mode = False
        self.current_step = 0
        self.current_layer = ""
        self.layer_order = []
        self.register_hooks()

    def register_hooks(self):
        for name, module in self.model.named_modules():
            module.register_forward_hook(self.forward_hook(name))
            module.register_backward_hook(self.backward_hook(name))
            self.layer_order.append(name)

    def forward_hook(self, name):
        def hook(module, input, output):
            self.current_state[name] = {
                'input': input,
                'output': output,
                'parameters': {param_name: param.data for param_name, param in module.named_parameters()}
            }
            self.current_step += 1
            self.current_layer = name
            if name in self.breakpoints:
                self.execution_paused = True
        return hook

    def backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.current_state[name]['grad_input'] = grad_input
            self.current_state[name]['grad_output'] = grad_output
            self.current_state[name]['grad_params'] = {param_name: param.grad for param_name, param in module.named_parameters()}
        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)