from typing import Optional
import torch

class ConditionalBreakpoint:
    def __init__(self, condition: Optional[str] = None, action: Optional[str] = None):
        self.condition = condition
        self.action = action

    def should_break(self, output) -> bool:
        if self.condition is None:
            return True
        try:
            return eval(self.condition, {'output': output, 'torch': torch})
        except Exception as e:
            print(f"Error evaluating breakpoint condition: {e}")
            return False

    def execute_action(self, model_wrapper, output):
        if self.action is not None:
            try:
                exec(self.action, {'self': model_wrapper, 'output': output, 'torch': torch})
            except Exception as e:
                print(f"Error executing breakpoint action: {e}")

    def __str__(self):
        return f"Condition: {self.condition or 'None'}, Action: {self.action or 'None'}"