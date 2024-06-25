from typing import Dict, List, Any

class BreakpointManager:
    def __init__(self, wrapped_model):
        self.wrapped_model = wrapped_model

    def set_breakpoint(self, layer_name: str, condition: str = None):
        if layer_name not in self.wrapped_model.breakpoints:
            self.wrapped_model.breakpoints[layer_name] = []
        self.wrapped_model.breakpoints[layer_name].append({'condition': condition})
        return f"Breakpoint set at {layer_name}" + (f" with condition: {condition}" if condition else "")

    def remove_breakpoint(self, layer_name: str):
        if layer_name in self.wrapped_model.breakpoints:
            del self.wrapped_model.breakpoints[layer_name]
            return f"Breakpoint removed at {layer_name}"
        return f"No breakpoint found at {layer_name}"

    def list_breakpoints(self) -> str:
        if not self.wrapped_model.breakpoints:
            return "No breakpoints set"
        return "\n".join([f"{layer}: {bps}" for layer, bps in self.wrapped_model.breakpoints.items()])