from core.model_wrapper import ModelWrapper

class ExecutionTracer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.execution_trace = []

    def trace(self):
        self.execution_trace = []
        self._register_hooks()
        return "Execution tracing enabled. Run the model to collect the trace."

    def _register_hooks(self):
        def hook(module, input, output):
            self.execution_trace.append({
                'layer_name': module.__class__.__name__,
                'input_shape': [tuple(i.shape) for i in input if hasattr(i, 'shape')],
                'output_shape': tuple(output.shape) if hasattr(output, 'shape') else None
            })

        for name, module in self.wrapped_model.model.named_modules():
            module.register_forward_hook(hook)

    def get_trace(self):
        return self.execution_trace

    def clear_trace(self):
        self.execution_trace = []
        return "Execution trace cleared."
    
    def traced_layers(self):
        return [step['layer_name'] for step in self.execution_trace]

    def summarize_trace(self):
        summary = []
        for idx, step in enumerate(self.execution_trace):
            summary.append(f"Step {idx}: {step['layer_name']} - Input: {step['input_shape']}, Output: {step['output_shape']}")
        return "\n".join(summary)