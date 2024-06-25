import torch

class StateModifier:
    def __init__(self, wrapped_model):
        self.wrapped_model = wrapped_model

    def modify_neuron(self, layer_name: str, neuron_index: int, value: float):
        if layer_name in self.wrapped_model.current_state:
            layer_state = self.wrapped_model.current_state[layer_name]
            layer_state['output'][..., neuron_index] = value
            print(f"Modified {layer_name}[{neuron_index}] to {value}")
        else:
            print(f"Layer {layer_name} not found")

    def modify_weight(self, layer_name: str, weight_name: str, indices: tuple, value: float):
        try:
            layer = self.wrapped_model.model.get_submodule(layer_name)
            if hasattr(layer, weight_name):
                weight = getattr(layer, weight_name)
                if isinstance(weight, torch.Tensor):
                    with torch.no_grad():
                        weight[indices] = value
                    print(f"Modified {layer_name}.{weight_name}{indices} to {value}")
                else:
                    print(f"Error: {weight_name} is not a tensor")
            else:
                print(f"Error: {weight_name} not found in {layer_name}")
        except AttributeError:
            print(f"Error: Layer {layer_name} not found")

    def reset_model(self):
        self.wrapped_model.model.load_state_dict(self.wrapped_model.initial_state_dict)
        print("Model weights reset to initial values")