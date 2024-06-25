import torch
from typing import List, Dict

def compare_model_outputs(models: List[torch.nn.Module], input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    outputs = {}
    for i, model in enumerate(models):
        outputs[f'model_{i}'] = model(input_ids).logits
    return outputs

def compute_output_difference(output1: torch.Tensor, output2: torch.Tensor) -> torch.Tensor:
    return torch.abs(output1 - output2)

def find_diverging_layers(model1: torch.nn.Module, model2: torch.nn.Module, input_ids: torch.Tensor, threshold: float = 1e-5):
    diverging_layers = []
    
    def compare_hook(name):
        def hook(module, input, output):
            output1 = output
            output2 = getattr(model2, name)(input[0])
            if torch.max(torch.abs(output1 - output2)) > threshold:
                diverging_layers.append(name)
        return hook
    
    hooks = []
    for name, module in model1.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Embedding)):
            hooks.append(module.register_forward_hook(compare_hook(name)))
    
    model1(input_ids)
    
    for hook in hooks:
        hook.remove()
    
    return diverging_layers