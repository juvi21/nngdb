import torch

def freeze_layers(model: torch.nn.Module, layers_to_freeze: list):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False
    print(f"Frozen layers: {layers_to_freeze}")

def replace_layer(model: torch.nn.Module, layer_name: str, new_layer: torch.nn.Module):
    name_parts = layer_name.split('.')
    module = model
    for part in name_parts[:-1]:
        module = getattr(module, part)
    setattr(module, name_parts[-1], new_layer)

def add_hook_to_layers(model: torch.nn.Module, hook_fn, layer_types=(torch.nn.Linear,)):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            hooks.append(module.register_forward_hook(hook_fn))
    return hooks