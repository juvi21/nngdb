import torch

def generate_random_input(input_shape: tuple, device: str = 'cpu'):
    return torch.randn(input_shape).to(device)

def generate_adversarial_input(model: torch.nn.Module, original_input: torch.Tensor, target_class: int, epsilon: float = 0.01, num_steps: int = 10):
    perturbed_input = original_input.clone().detach().requires_grad_(True)
    
    for _ in range(num_steps):
        output = model(perturbed_input)
        loss = -output[0, target_class]
        loss.backward()
        
        with torch.no_grad():
            perturbed_input += epsilon * perturbed_input.grad.sign()
            perturbed_input.clamp_(0, 1)  # Assuming input values are between 0 and 1
        
        perturbed_input.grad.zero_()
    
    return perturbed_input.detach()