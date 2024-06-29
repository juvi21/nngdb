from core.model_wrapper import ModelWrapper
import torch


# tracing/execution_tracer.py

class ExecutionTracer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.execution_trace = []
        self.token_trace = {}

    def trace(self):
        self.execution_trace = []
        self.token_trace = {}
        self._register_hooks()
        return "Execution tracing enabled. Run the model to collect the trace."

    def _register_hooks(self):
        def hook(module, input, output):
            layer_name = next(name for name, mod in self.wrapped_model.model.named_modules() if mod is module)
            if isinstance(output, torch.Tensor):
                output_data = output.detach()
            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                output_data = output[0].detach()
            else:
                output_data = None

            self.execution_trace.append({
                'layer_name': module.__class__.__name__,
                'layer_full_name': layer_name,
                'input_shape': [tuple(i.shape) for i in input if isinstance(i, torch.Tensor)],
                'output_shape': tuple(output_data.shape) if output_data is not None else None,
                'output': output_data
            })

        for name, module in self.wrapped_model.model.named_modules():
            module.register_forward_hook(hook)

    def trace_token(self, token_id: int, position: int):
        print(f"Tracing token ID: {token_id} at position: {position}")  # Debug print
        if token_id not in self.token_trace:
            self.token_trace[token_id] = []
        for step in self.execution_trace:
            if step['output'] is not None:
                if step['output'].dim() >= 2 and step['output'].size(1) > position:
                    token_output = step['output'][:, position]
                    self.token_trace[token_id].append({
                        'layer_name': step['layer_full_name'],
                        'output': token_output,
                        'position': position
                    })
        print(f"Token trace length: {len(self.token_trace[token_id])}")  # Debug print
        print(f"Layers: {[step['layer_name'] for step in self.token_trace[token_id]]}")  # Debug print

    def get_token_trace(self, token_id: int):
        return self.token_trace.get(token_id, [])

    def clear_trace(self):
        self.execution_trace = []
        self.token_trace = {}
        return "Execution trace cleared."

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