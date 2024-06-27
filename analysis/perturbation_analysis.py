import torch
from core.model_wrapper import ModelWrapper

class PerturbationAnalyzer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def analyze_input_perturbation(self, input_tensor: torch.Tensor, perturbation_scale: float = 0.1, num_samples: int = 10):
        original_output = self.wrapped_model.model(input_tensor)
        
        perturbed_outputs = []
        for _ in range(num_samples):
            perturbation = torch.randn_like(input_tensor) * perturbation_scale
            perturbed_input = input_tensor + perturbation
            perturbed_output = self.wrapped_model.model(perturbed_input)
            perturbed_outputs.append(perturbed_output)

        output_diff = torch.stack([output - original_output for output in perturbed_outputs])
        
        analysis = {
            "mean_output_diff": output_diff.abs().mean().item(),
            "std_output_diff": output_diff.std().item(),
            "max_output_diff": output_diff.abs().max().item(),
        }

        return analysis

    def analyze_weight_perturbation(self, layer_name: str, perturbation_scale: float = 0.1, num_samples: int = 10):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        original_weights = {name: param.clone() for name, param in layer.named_parameters() if 'weight' in name}
        original_output = self.wrapped_model.model(self.wrapped_model.current_state['input'])

        perturbed_outputs = []
        for _ in range(num_samples):
            with torch.no_grad():
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        perturbation = torch.randn_like(param) * perturbation_scale
                        param.add_(perturbation)
                
                perturbed_output = self.wrapped_model.model(self.wrapped_model.current_state['input'])
                perturbed_outputs.append(perturbed_output)

                # Restore original weights
                for name, original_weight in original_weights.items():
                    getattr(layer, name).copy_(original_weight)

        output_diff = torch.stack([output - original_output for output in perturbed_outputs])

        analysis = {
            "mean_output_diff": output_diff.abs().mean().item(),
            "std_output_diff": output_diff.std().item(),
            "max_output_diff": output_diff.abs().max().item(),
        }

        return analysis

    def compute_input_gradient(self, input_tensor: torch.Tensor):
        input_tensor.requires_grad_(True)
        output = self.wrapped_model.model(input_tensor)
        
        if isinstance(output, torch.Tensor):
            loss = output.sum()
        elif isinstance(output, tuple):
            loss = output[0].sum()  # Assuming the first element is the main output
        else:
            return "Unexpected output type. Unable to compute input gradient."

        loss.backward()

        input_gradient = input_tensor.grad

        return {
            "input_gradient_norm": input_gradient.norm().item(),
            "input_gradient_mean": input_gradient.abs().mean().item(),
            "input_gradient_std": input_gradient.std().item(),
        }

    def compute_saliency_map(self, input_tensor: torch.Tensor):
        input_tensor.requires_grad_(True)
        output = self.wrapped_model.model(input_tensor)
        
        if isinstance(output, torch.Tensor):
            loss = output.sum()
        elif isinstance(output, tuple):
            loss = output[0].sum()
        else:
            return "Unexpected output type. Unable to compute saliency map."

        loss.backward()

        saliency_map = input_tensor.grad.abs()
        
        return {
            "saliency_map": saliency_map,
            "max_saliency": saliency_map.max().item(),
            "mean_saliency": saliency_map.mean().item(),
        }