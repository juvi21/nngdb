import torch
from typing import Dict, Any
from .model_wrapper import NNGDBWrapper
from .breakpoint_manager import BreakpointManager
from .state_inspector import StateInspector
from .state_modifier import StateModifier
from .execution_engine import ExecutionEngine
from .profiler import Profiler

class NNGDB:
    def __init__(self, model: torch.nn.Module):
        self.wrapped_model = NNGDBWrapper(model)
        self.breakpoint_manager = BreakpointManager(self.wrapped_model)
        self.state_inspector = StateInspector(self.wrapped_model)
        self.state_modifier = StateModifier(self.wrapped_model)
        self.execution_engine = ExecutionEngine(self.wrapped_model)
        self.profiler = Profiler()
        self.context: Dict[str, Any] = {}

    def run(self, input_data: torch.Tensor):
        return self.execution_engine.run(input_data)

    def set_context(self, key: str, value: Any):
        self.context[key] = value

    def get_context(self, key: str) -> Any:
        return self.context.get(key)