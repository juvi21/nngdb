import torch
import torch.nn.functional as F
import numpy as np

class ExplainabilityTechniques:
    @staticmethod
    def compute_saliency_map(model: torch.nn.Module, inputs: torch.Tensor, target_class: int):
        model.eval()
        inputs.requires_grad_(True)
        
        outputs = model(inputs)
        score = outputs[:, target_class].sum()
        score.backward()
        
        saliency, _ = torch.max(inputs.grad.data.abs(), dim=1)
        return saliency

    @staticmethod
    def integrated_gradients(model: torch.nn.Module, inputs: torch.Tensor, target_class: int, steps: int = 100):
        model.eval()
        inputs.requires_grad_(True)
        
        baseline = torch.zeros_like(inputs)
        scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]
        grads = []
        
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad_(True)
            outputs = model(scaled_input)
            score = outputs[:, target_class].sum()
            score.backward()
            grads.append(scaled_input.grad.clone())
            scaled_input.grad.zero_()
        
        avg_grads = torch.cat(grads).mean(dim=0)
        integrated_grad = (inputs - baseline) * avg_grads
        return integrated_grad

    @staticmethod
    def occlusion_sensitivity(model: torch.nn.Module, inputs: torch.Tensor, target_class: int, window_size: int = 8):
        model.eval()
        batch_size, channels, height, width = inputs.shape
        occlusion_map = torch.zeros((height, width))
        
        original_score = model(inputs)[:, target_class].item()
        
        for i in range(0, height, window_size):
            for j in range(0, width, window_size):
                occluded_input = inputs.clone()
                occluded_input[:, :, i:i+window_size, j:j+window_size] = 0
                occluded_score = model(occluded_input)[:, target_class].item()
                occlusion_map[i:i+window_size, j:j+window_size] = original_score - occluded_score
        
        return occlusion_map

    @staticmethod
    def grad_cam(model: torch.nn.Module, inputs: torch.Tensor, target_class: int, layer_name: str):
        model.eval()
        
        # Get the specified layer
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = module
                break
        else:
            raise ValueError(f"Layer {layer_name} not found in the model")
        
        # Hook for getting layer output and gradients
        layer_output = None
        layer_grad = None
        
        def save_output_and_grad(module, input, output):
            nonlocal layer_output, layer_grad
            layer_output = output
            output.register_hook(lambda grad: setattr(grad, 'grad', grad))
        
        handle = target_layer.register_forward_hook(save_output_and_grad)
        
        # Forward pass
        model_output = model(inputs)
        model_output[0, target_class].backward()
        
        handle.remove()
        
        # Compute GradCAM
        gradients = layer_output.grad
        pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(layer_output * pooled_gradients, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=inputs.shape[2:], mode='bilinear', align_corners=False)
        cam = cam / cam.max()
        
        return cam.squeeze()