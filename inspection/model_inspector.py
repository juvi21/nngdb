from core.model_wrapper import ModelWrapper

class ModelInspector:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def inspect(self):
        model_info = {
            "model_type": type(self.wrapped_model.model).__name__,
            "num_parameters": sum(p.numel() for p in self.wrapped_model.model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in self.wrapped_model.model.parameters() if p.requires_grad),
            "layers": self._get_layers_info()
        }
        return model_info

    def _get_layers_info(self):
        layers_info = {}
        for name, module in self.wrapped_model.model.named_modules():
            if not list(module.children()):  # Only leaf modules
                layers_info[name] = {
                    "type": type(module).__name__,
                    "parameters": {
                        param_name: {
                            "shape": param.shape,
                            "requires_grad": param.requires_grad
                        } for param_name, param in module.named_parameters()
                    }
                }
        return layers_info