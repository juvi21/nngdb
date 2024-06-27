import torch
import torch.nn.functional as F

class InterpretabilityMetrics:
    @staticmethod
    def compute_activation_stability(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, layer_name: str):
        activations = []
        
        def hook(module, input, output):
            activations.append(output.detach())
        
        for name, module in model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook)
                break
        else:
            return f"Layer '{layer_name}' not found"
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0]
                model(inputs)
        
        handle.remove()
        
        activations = torch.cat(activations, dim=0)
        stability = torch.std(activations, dim=0).mean().item()
        
        return f"Activation stability for layer '{layer_name}': {stability:.4f}"

    @staticmethod
    def compute_loss_sensitivity(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-4):
        model.eval()
        inputs.requires_grad_(True)
        
        original_loss = F.cross_entropy(model(inputs), targets)
        original_loss.backward()
        
        input_grad = inputs.grad.data
        
        perturbed_inputs = inputs + epsilon * input_grad.sign()
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
        with torch.no_grad():
            perturbed_loss = F.cross_entropy(model(perturbed_inputs), targets)
        
        sensitivity = (perturbed_loss - original_loss) / epsilon
        
        return f"Loss sensitivity: {sensitivity.item():.4f}"

    @staticmethod
    def compute_decision_boundary_distance(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor, num_steps: int = 100, step_size: float = 0.01):
        model.eval()
        inputs.requires_grad_(True)
        
        for _ in range(num_steps):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            with torch.no_grad():
                inputs -= step_size * inputs.grad.sign()
                inputs.clamp_(0, 1)
            
            inputs.grad.zero_()
            
            predicted = outputs.argmax(dim=1)
            if (predicted != targets).any():
                break
        
        distance = (inputs - inputs.data).norm(dim=1).mean()
        return f"Average distance to decision boundary: {distance.item():.4f}"