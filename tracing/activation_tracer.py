import torch
from core.model_wrapper import ModelWrapper

class ActivationTracer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.activation_trace = {}

    def trace(self):
        def hook(module, input, output):
            self.activation_trace[module.__class__.__name__] = output.detach()

        for name, module in self.wrapped_model.model.named_modules():
            module.register_forward_hook(hook)

    def _register_hooks(self):
        def hook(module, input, output):
            self.activation_trace[module.__class__.__name__] = output.detach() if isinstance(output, torch.Tensor) else output

        for name, module in self.wrapped_model.model.named_modules():
            module.register_forward_hook(hook)

    def get_activation(self, layer_name: str):
        return self.activation_trace.get(layer_name, f"No activation recorded for layer '{layer_name}'")

    def get_all_activations(self):
        return self.activation_trace

    def clear_trace(self):
        self.activation_trace = {}
        return "Activation trace cleared."

    def summarize_trace(self):
        summary = []
        for layer_name, activation in self.activation_trace.items():
            if isinstance(activation, torch.Tensor):
                summary.append(f"{layer_name}: shape={activation.shape}, mean={activation.mean().item():.4f}, std={activation.std().item():.4f}")
            else:
                summary.append(f"{layer_name}: {type(activation)}")
        return "\n".join(summary)