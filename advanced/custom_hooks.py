import torch
from typing import Callable, Dict, Any

class CustomHookManager:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.hooks: Dict[str, Any] = {}

    def register_forward_hook(self, module_name: str, hook: Callable, name: str):
        module = dict(self.model.named_modules())[module_name]
        handle = module.register_forward_hook(hook)
        self.hooks[name] = handle
        return f"Forward hook '{name}' registered for module '{module_name}'"

    def register_backward_hook(self, module_name: str, hook: Callable, name: str):
        module = dict(self.model.named_modules())[module_name]
        handle = module.register_full_backward_hook(hook)
        self.hooks[name] = handle
        return f"Backward hook '{name}' registered for module '{module_name}'"

    def remove_hook(self, name: str):
        if name in self.hooks:
            self.hooks[name].remove()
            del self.hooks[name]
            return f"Hook '{name}' removed"
        return f"Hook '{name}' not found"

    def clear_all_hooks(self):
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
        return "All hooks cleared"

    def list_hooks(self):
        return "\n".join(f"{name}: {type(hook).__name__}" for name, hook in self.hooks.items())