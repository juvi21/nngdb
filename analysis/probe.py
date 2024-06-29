# nngdb/analysis/probe.py

from typing import Callable, Dict, Any, List, Optional
import torch
import cloudpickle

class SaveContext:
    def __init__(self):
        self.data: Dict[str, Any] = {}

    def __setattr__(self, key: str, value: Any):
        if key == 'data':
            super().__setattr__(key, value)
        else:
            self.data[key] = value

    def __getattr__(self, key: str):
        return self.data.get(key)

class ProbePoint:
    def __init__(self, name: str):
        self.name = name
        self.hooks: List[Callable] = []

    def probe(self, hook: Callable):
        self.hooks.append(hook)
        return self

    def clear(self):
        self.hooks.clear()

class ProbeManager:
    def __init__(self):
        self.probe_points: Dict[str, ProbePoint] = {}
        self.recordings: Dict[str, SaveContext] = {}

    def register_probe_point(self, name: str):
        if name not in self.probe_points:
            self.probe_points[name] = ProbePoint(name)

    def get_probe_point(self, name: str) -> Optional[ProbePoint]:
        return self.probe_points.get(name)

    def forward_hook(self, module, input, output):
        module_name = next(name for name, mod in self.model.named_modules() if mod is module)
        if module_name in self.probe_points:
            save_ctx = SaveContext()
            self.recordings[module_name] = save_ctx
            for hook in self.probe_points[module_name].hooks:
                result = hook(save_ctx, output)
                if result is not None:
                    output = result
        return output

    def backward_hook(self, module, grad_input, grad_output):
        module_name = next(name for name, mod in self.model.named_modules() if mod is module)
        if module_name in self.probe_points:
            save_ctx = SaveContext()
            self.recordings[module_name] = save_ctx
            for hook in self.probe_points[module_name].hooks:
                result = hook(save_ctx, grad_input, grad_output)
                if result is not None:
                    grad_input, grad_output = result
        return grad_input, grad_output

    def clear_recordings(self):
        self.recordings.clear()

def probe_decorator(func):
    def wrapper(self, *args, **kwargs):
        probes = kwargs.pop('probes', [])
        forward_hooks = []
        backward_hooks = []
        
        for probe in probes:
            module = dict(self.model.named_modules())[probe.name]
            forward_hooks.append(module.register_forward_hook(self.probe_manager.forward_hook))
            backward_hooks.append(module.register_full_backward_hook(self.probe_manager.backward_hook))
        
        try:
            result = func(self, *args, **kwargs)
        finally:
            for hook in forward_hooks + backward_hooks:
                hook.remove()
        
        return result
    return wrapper