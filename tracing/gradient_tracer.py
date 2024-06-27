import torch
from core.model_wrapper import ModelWrapper

class GradientTracer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.gradient_trace = {}

    def trace(self):
        self.gradient_trace = {}
        self._register_hooks()
        return "Gradient tracing enabled. Run backward pass to collect gradients."

    def _register_hooks(self):
        def hook(module, grad_input, grad_output):
            self.gradient_trace[module.__class__.__name__] = {
                'grad_input': [g.detach() if isinstance(g, torch.Tensor) else g for g in grad_input],
                'grad_output': [g.detach() if isinstance(g, torch.Tensor) else g for g in grad_output]
            }

        for name, module in self.wrapped_model.model.named_modules():
            module.register_full_backward_hook(hook)

    def get_gradient(self, layer_name: str):
        return self.gradient_trace.get(layer_name, f"No gradient recorded for layer '{layer_name}'")

    def get_all_gradients(self):
        return self.gradient_trace

    def clear_trace(self):
        self.gradient_trace = {}
        return "Gradient trace cleared."

    def summarize_trace(self):
        summary = []
        for layer_name, grads in self.gradient_trace.items():
            summary.append(f"Layer: {layer_name}")
            for grad_type in ['grad_input', 'grad_output']:
                for idx, grad in enumerate(grads[grad_type]):
                    if isinstance(grad, torch.Tensor):
                        summary.append(f"  {grad_type}[{idx}]: shape={grad.shape}, mean={grad.mean().item():.4f}, std={grad.std().item():.4f}")
                    else:
                        summary.append(f"  {grad_type}[{idx}]: {type(grad)}")
        return "\n".join(summary)