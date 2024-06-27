import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import copy

class ModelWrapper(nn.Module):
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
        self.attention_weights = {}
        self.initial_state_dict = copy.deepcopy(model.state_dict())
        self.modified_weights = {}
        self.activation_hooks = {}
        self.register_hooks()

    def register_hooks(self):
        def attention_hook(module, input, output):
            self.attention_weights[module.__class__.__name__] = output[1] if isinstance(output, tuple) else output

        for name, module in self.model.named_modules():
            if "self_attn" in name:
                module.register_forward_hook(attention_hook)
            module.register_forward_hook(self.forward_hook(name))
            module.register_full_backward_hook(self.backward_hook(name))
            self.layer_order.append(name)

    def forward_hook(self, name):
        def hook(module, input, output):
            self.current_state[name] = {
                'input': input,
                'output': output,
                'parameters': {param_name: param.data for param_name, param in module.named_parameters()}
            }
            self.current_layer = name
            if name in self.breakpoints:
                for breakpoint in self.breakpoints[name]:
                    if breakpoint['condition'] is None or self._evaluate_condition(breakpoint['condition'], output):
                        self.execution_paused = True
                        print(f"Breakpoint hit at layer: {name}")
                        if breakpoint['condition']:
                            print(f"Condition satisfied: {breakpoint['condition']}")
                        return
        return hook

    def backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.current_state[name]['grad_input'] = grad_input
            self.current_state[name]['grad_output'] = grad_output
            self.current_state[name]['grad_params'] = {param_name: param.grad for param_name, param in module.named_parameters()}
        return hook
    
    def _evaluate_condition(self, condition: str, output):
        try:
            return eval(condition, {'output': output, 'torch': torch})
        except Exception as e:
            print(f"Error evaluating breakpoint condition: {str(e)}")
            return False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def get_parameter(self, param_name: str) -> Optional[nn.Parameter]:
        return dict(self.model.named_parameters()).get(param_name)

    def set_breakpoint(self, layer_name: str, condition: Optional[str] = None, action: Optional[str] = None):
        if layer_name not in self.breakpoints:
            self.breakpoints[layer_name] = []
        self.breakpoints[layer_name].append({'condition': condition, 'action': action})

    def remove_breakpoint(self, layer_name: str):
        if layer_name in self.breakpoints:
            del self.breakpoints[layer_name]

    def clear_all_breakpoints(self):
        self.breakpoints.clear()

    def reset_to_initial_state(self):
        self.model.load_state_dict(self.initial_state_dict)
        print("Model reset to initial state.")

    def step_to_layer(self, target_layer: str):
        if not self.current_state:
            return "Error: No execution state. Run the model first."
        
        def find_layer(model, target):
            for name, child in model.named_children():
                if name == target:
                    return child
                result = find_layer(child, target)
                if result is not None:
                    return result
            return None

        target_module = find_layer(self.model, target_layer)
        if target_module is None:
            return f"Error: Layer {target_layer} not found"
    
    def get_attention_weights(self, layer_name: str):
        return self.attention_weights.get(layer_name)
    
    def modify_weight(self, layer_name: str, weight_name: str, indices, value):
        layer = self.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, weight_name):
            return f"Weight '{weight_name}' not found in layer '{layer_name}'."

        weight = getattr(layer, weight_name)
        if not isinstance(weight, torch.Tensor):
            return f"'{weight_name}' is not a tensor in layer '{layer_name}'."

        try:
            with torch.no_grad():
                original_value = weight[indices].clone()
                new_weight = weight.clone()
                new_weight[indices] = value
                setattr(layer, weight_name, nn.Parameter(new_weight))
                
            self.modified_weights[(layer_name, weight_name, indices)] = original_value
            return f"Weight at {layer_name}.{weight_name}{indices} modified to {value}"
        except Exception as e:
            return f"Error modifying weight: {str(e)}"

    def reset_modified_weights(self):
        with torch.no_grad():
            for (layer_name, weight_name, indices), original_value in self.modified_weights.items():
                layer = self.get_layer(layer_name)
                weight = getattr(layer, weight_name)
                new_weight = weight.clone()
                new_weight[indices] = original_value
                setattr(layer, weight_name, nn.Parameter(new_weight))
        self.modified_weights.clear()
        return "All modified weights have been reset."
    

    def modify_activation(self, layer_name: str, function_str: str):
        layer = self.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        try:
            modification_function = eval(f"lambda x: {function_str}")
        except Exception as e:
            return f"Error in function definition: {str(e)}"

        def hook(module, input, output):
            return modification_function(output)

        # Remove existing hook if there is one
        if layer_name in self.activation_hooks:
            self.activation_hooks[layer_name].remove()

        handle = layer.register_forward_hook(hook)
        self.activation_hooks[layer_name] = handle

        return f"Activation modification hook set for layer '{layer_name}'"

    def clear_activation_modifications(self):
        for handle in self.activation_hooks.values():
            handle.remove()
        self.activation_hooks.clear()
        return "All activation modifications cleared"

    def get_layer(self, layer_name: str) -> Optional[nn.Module]:
        parts = layer_name.split('.')
        current = self.model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current



    
    