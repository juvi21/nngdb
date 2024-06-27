from typing import Dict, List, Optional
from .conditional_breakpoint import ConditionalBreakpoint
from core.model_wrapper import ModelWrapper

class BreakpointManager:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.breakpoints: Dict[str, List[ConditionalBreakpoint]] = {}

    def set_breakpoint(self, layer_name: str, condition: str = None):
        if layer_name not in self.breakpoints:
            self.breakpoints[layer_name] = []
        self.breakpoints[layer_name].append(condition)
        self.wrapped_model.set_breakpoint(layer_name, condition)
        return f"Breakpoint set at {layer_name}" + (f" with condition: {condition}" if condition else "")
    
    def remove_breakpoint(self, layer_name: str, index: int = -1):
        if layer_name in self.breakpoints:
            if 0 <= index < len(self.breakpoints[layer_name]):
                removed = self.breakpoints[layer_name].pop(index)
                if not self.breakpoints[layer_name]:
                    del self.breakpoints[layer_name]
                # Remove the breakpoint from the model wrapper
                self.wrapped_model.remove_breakpoint(layer_name)
                return f"Breakpoint removed at {layer_name}, index {index}"
            elif index == -1 and self.breakpoints[layer_name]:
                del self.breakpoints[layer_name]
                # Remove all breakpoints for this layer from the model wrapper
                self.wrapped_model.remove_breakpoint(layer_name)
                return f"All breakpoints removed at {layer_name}"
        return f"No breakpoint found at {layer_name}" + (f", index {index}" if index != -1 else "")

    def list_breakpoints(self) -> str:
        if not self.breakpoints:
            return "No breakpoints set"
        
        breakpoint_list = []
        for layer, bps in self.breakpoints.items():
            for idx, bp in enumerate(bps):
                breakpoint_list.append(f"{layer}[{idx}]: {bp}")
        
        return "\n".join(breakpoint_list)

    def clear_all_breakpoints(self):
        self.breakpoints.clear()
        self.wrapped_model.clear_all_breakpoints()
        return "All breakpoints cleared"

    def hit_breakpoint(self, layer_name: str, output):
        if layer_name in self.breakpoints:
            for bp in self.breakpoints[layer_name]:
                if bp.should_break(output):
                    print(f"Breakpoint hit at {layer_name}")
                    bp.execute_action(self.wrapped_model, output)
                    return True
        return False