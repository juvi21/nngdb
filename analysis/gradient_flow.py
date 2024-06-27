import torch
from core.model_wrapper import ModelWrapper

class GradientFlowAnalyzer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def analyze(self):
        gradients = {}
        for name, param in self.wrapped_model.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients[name] = {
                    'mean': param.grad.abs().mean().item(),
                    'std': param.grad.std().item(),
                    'max': param.grad.abs().max().item(),
                    'norm': param.grad.norm().item()
                }

        return gradients

    def detect_vanishing_exploding_gradients(self, threshold=1e-4):
        vanishing = []
        exploding = []
        for name, param in self.wrapped_model.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm < threshold:
                    vanishing.append((name, grad_norm))
                elif grad_norm > 1/threshold:
                    exploding.append((name, grad_norm))
        
        return {
            'vanishing': vanishing,
            'exploding': exploding
        }

    def compute_layer_gradients(self):
        layer_gradients = {}
        for name, module in self.wrapped_model.model.named_modules():
            if list(module.children()):  # skip container modules
                continue
            layer_grad = torch.cat([p.grad.view(-1) for p in module.parameters() if p.grad is not None])
            layer_gradients[name] = {
                'mean': layer_grad.abs().mean().item(),
                'std': layer_grad.std().item(),
                'max': layer_grad.abs().max().item(),
                'norm': layer_grad.norm().item()
            }
        return layer_gradients