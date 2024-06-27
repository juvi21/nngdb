from core.model_wrapper import ModelWrapper

class VariableInspector:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def inspect(self, variable_name: str):
        parts = variable_name.split('.')
        current = self.wrapped_model.model

        try:
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return f"Variable '{variable_name}' not found."

            return self._analyze_variable(current)
        except Exception as e:
            return f"Error inspecting variable '{variable_name}': {str(e)}"

    def _analyze_variable(self, variable):
        import torch

        if isinstance(variable, torch.Tensor):
            return {
                "type": "Tensor",
                "shape": variable.shape,
                "dtype": str(variable.dtype),
                "device": str(variable.device),
                "requires_grad": variable.requires_grad,
                "is_leaf": variable.is_leaf,
                "mean": variable.mean().item(),
                "std": variable.std().item(),
                "min": variable.min().item(),
                "max": variable.max().item(),
            }
        elif isinstance(variable, torch.nn.Parameter):
            return {
                "type": "Parameter",
                "shape": variable.shape,
                "dtype": str(variable.dtype),
                "device": str(variable.device),
                "requires_grad": variable.requires_grad,
                "is_leaf": variable.is_leaf,
                "mean": variable.mean().item(),
                "std": variable.std().item(),
                "min": variable.min().item(),
                "max": variable.max().item(),
            }
        else:
            return {
                "type": type(variable).__name__,
                "value": str(variable),
            }

    def get_variable(self, variable_name: str):
        parts = variable_name.split('.')
        current = self.wrapped_model.model

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return f"Variable '{variable_name}' not found."

        return current