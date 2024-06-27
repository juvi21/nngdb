from core.model_wrapper import ModelWrapper
import torch

class HyperparameterModifier:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def modify_learning_rate(self, optimizer, new_lr: float):
        if not hasattr(self.wrapped_model, 'optimizer'):
            return "No optimizer found. Please set an optimizer for the model first."

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        return f"Learning rate modified to {new_lr}"

    def modify_weight_decay(self, optimizer, new_weight_decay: float):
        if not hasattr(self.wrapped_model, 'optimizer'):
            return "No optimizer found. Please set an optimizer for the model first."

        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = new_weight_decay

        return f"Weight decay modified to {new_weight_decay}"

    def modify_dropout_rate(self, dropout_rate: float):
        modified_layers = []
        for name, module in self.wrapped_model.model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate
                modified_layers.append(name)

        if modified_layers:
            return f"Dropout rate modified to {dropout_rate} for layers: {', '.join(modified_layers)}"
        else:
            return "No dropout layers found in the model."

    def freeze_layers(self, layer_names):
        frozen_layers = []
        for name, param in self.wrapped_model.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                frozen_layers.append(name)

        if frozen_layers:
            return f"Layers frozen: {', '.join(frozen_layers)}"
        else:
            return "No layers matched the provided names."

    def unfreeze_layers(self, layer_names):
        unfrozen_layers = []
        for name, param in self.wrapped_model.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                unfrozen_layers.append(name)

        if unfrozen_layers:
            return f"Layers unfrozen: {', '.join(unfrozen_layers)}"
        else:
            return "No layers matched the provided names."