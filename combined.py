# ./__init__.py
from .core import NNGDB
from .cli import NNGDBREPL
from . import utils

__version__ = "0.1.0"

__all__ = ['NNGDB', 'NNGDBREPL', 'utils']

# ./nngdb_server.py
import socket
import threading
import cloudpickle
import struct
from core.debugger import NNGDB
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class NNGDBServer:
    def __init__(self, model_name, device, port=5000):
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Initializing NNGDB")
        self.debugger = NNGDB(model, model_name, device)
        self.debugger.set_context('tokenizer', tokenizer)
        self.debugger.set_context('device', device)

        self.port = port
        self.clients = {}
        self.lock = threading.Lock()

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(5)
        print(f"Server started on port {self.port}")

        while True:
            client_socket, addr = server_socket.accept()
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket, addr))
            client_thread.start()

    def handle_client(self, client_socket, addr):
        print(f"New connection from {addr}")
        while True:
            try:
                data = self.recv_msg(client_socket)
                if not data:
                    break
                command = cloudpickle.loads(data)
                result = self.execute_command(command)
                self.send_msg(client_socket, cloudpickle.dumps(result))
            except Exception as e:
                print(f"Error handling client {addr}: {e}")
                break
        print(f"Connection from {addr} closed")
        client_socket.close()

    def execute_command(self, command):
        with self.lock:
            method = getattr(self.debugger, command['method'])
            return method(*command['args'], **command['kwargs'])

    def send_msg(self, sock, msg):
        msg = struct.pack('>I', len(msg)) + msg
        sock.sendall(msg)

    def recv_msg(self, sock):
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        return self.recvall(sock, msglen)

    def recvall(self, sock, n):
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch NNGDB Server")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()

    server = NNGDBServer(args.model, args.device, args.port)
    server.start()

# ./nngdb.py
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import NNGDB
from cli import NNGDBREPL

def main():
    parser = argparse.ArgumentParser(description="Neural Network GDB (NNGDB)")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--server", type=str, help="Connect to a running server (host:port)")
    args = parser.parse_args()

    if args.server:
        host, port = args.server.split(':')
        debugger = NNGDB(None, args.model, args.device)
        debugger.connect(host, int(port))
        print(f"Connected to NNGDB server at {host}:{port}")
    else:
        print(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        print(f"Initializing NNGDB")
        debugger = NNGDB(model, args.model, args.device)
        debugger.set_context('tokenizer', tokenizer)
        debugger.set_context('device', args.device)

    print("Starting NNGDB REPL")
    repl = NNGDBREPL(debugger)
    repl.run()

if __name__ == "__main__":
    main()

# ./helper.py
import os

def combine_files(directory):
    combined_content = ""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                combined_content += f"# {file_path}\n{content}\n\n"
    return combined_content

# Use the function
directory = '.'  # current directory
combined_content = combine_files(directory)

# Write the combined content to a new file
with open('combined.py', 'w') as f:
    f.write(combined_content)

# ./experiments/__init__.py
from .experiment_manager import ExperimentManager



__all__ = ['ExperimentManager']

# ./experiments/experiment_manager.py
import copy

class ExperimentManager:
    def __init__(self, model):
        self.base_model = model
        self.experiments = {}
        self.current_experiment = None

    def create_experiment(self, name):
        if name in self.experiments:
            return f"Experiment '{name}' already exists."
        self.experiments[name] = copy.deepcopy(self.base_model.state_dict())
        return f"Experiment '{name}' created."

    def switch_experiment(self, name):
        if name not in self.experiments:
            return f"Experiment '{name}' does not exist."
        self.base_model.load_state_dict(self.experiments[name])
        self.current_experiment = name
        return f"Switched to experiment '{name}'."

    def list_experiments(self):
        return list(self.experiments.keys())

    def delete_experiment(self, name):
        if name not in self.experiments:
            return f"Experiment '{name}' does not exist."
        del self.experiments[name]
        if self.current_experiment == name:
            self.current_experiment = None
        return f"Experiment '{name}' deleted."

    def get_current_experiment(self):
        return self.current_experiment

# ./breakpoints/__init__.py
from .breakpoint_manager import BreakpointManager
from .conditional_breakpoint import ConditionalBreakpoint

__all__ = ['BreakpointManager', 'ConditionalBreakpoint']

# ./breakpoints/breakpoint_manager.py
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

# ./breakpoints/conditional_breakpoint.py
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

# ./core/__init__.py
from .debugger import NNGDB
from .model_wrapper import ModelWrapper
from .execution_engine import ExecutionEngine
from .undo_manager import UndoManager

__all__ = ['NNGDB', 'ModelWrapper', 'ExecutionEngine', 'UndoManager']

# ./core/execution_engine.py
import torch
from .model_wrapper import ModelWrapper

class ExecutionEngine:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.current_layer_index = 0
        self.last_input = None

    def run(self, input_data: torch.Tensor):
        self.wrapped_model.execution_paused = False
        self.current_layer_index = 0
        self.last_input = input_data
        return self.wrapped_model(input_data)

    def continue_execution(self):
        if self.last_input is None:
            return "Error: No previous execution. Run the model first."
        self.wrapped_model.execution_paused = False
        print("Continuing execution...")
        return self.wrapped_model(self.last_input)

    def step(self, num_steps: int = 1):
        if not self.wrapped_model.layer_order:
            return "Error: Model layers not initialized. Run the model first."
        
        if self.current_layer_index is None:
            return "Error: No previous execution. Run the model first."
        
        target_index = min(self.current_layer_index + num_steps, len(self.wrapped_model.layer_order) - 1)
        target_layer = self.wrapped_model.layer_order[target_index]
        self.current_layer_index = target_index
        print(f"Stepping {num_steps} layer(s) to {target_layer}")
        return self.wrapped_model.step_to_layer(target_layer)

    def run_backward(self, loss: torch.Tensor):
        loss.backward()
        print("Backward pass completed. Gradients computed.")

    def reset(self):
        self.wrapped_model.reset_to_initial_state()
        self.wrapped_model.current_step = 0
        self.wrapped_model.current_layer = ""
        self.wrapped_model.execution_paused = False
        self.wrapped_model.step_mode = False
        print("Execution engine reset.")

# ./core/undo_manager.py
class UndoManager:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
        self.current_index = -1

    def add_state(self, state):
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.current_index = len(self.history) - 1

    def undo(self):
        if self.current_index > 0:
            self.current_index -= 1
            return self.history[self.current_index]
        return None

    def redo(self):
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            return self.history[self.current_index]
        return None

# ./core/model_wrapper.py
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import copy

class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.initial_state_dict = copy.deepcopy(model.state_dict())
        self.breakpoints: Dict[str, List[Dict[str, Any]]] = {}
        self.current_state: Dict[str, Any] = {}
        self.execution_paused = False
        self.step_mode = False
        self.current_step = 0
        self.current_layer = ""
        self.layer_order = []
        self.attention_weights = {}
        self.initial_state_dict = copy.deepcopy(model.state_dict())
        self.modified_weights = {}
        self.activation_hooks = {}
        self.register_hooks()

    def register_hooks(self):
        def attention_hook(module, input, output):
            self.attention_weights[module.__class__.__name__] = output[1] if isinstance(output, tuple) else output

        for name, module in self.model.named_modules():
            if "self_attn" in name:
                module.register_forward_hook(attention_hook)
            module.register_forward_hook(self.forward_hook(name))
            module.register_full_backward_hook(self.backward_hook(name))
            self.layer_order.append(name)

    def forward_hook(self, name):
        def hook(module, input, output):
            self.current_state[name] = {
                'input': input,
                'output': output,
                'parameters': {param_name: param.data for param_name, param in module.named_parameters()}
            }
            self.current_layer = name
            if name in self.breakpoints:
                for breakpoint in self.breakpoints[name]:
                    if breakpoint['condition'] is None or self._evaluate_condition(breakpoint['condition'], output):
                        self.execution_paused = True
                        print(f"Breakpoint hit at layer: {name}")
                        if breakpoint['condition']:
                            print(f"Condition satisfied: {breakpoint['condition']}")
                        return
        return hook

    def backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.current_state[name]['grad_input'] = grad_input
            self.current_state[name]['grad_output'] = grad_output
            self.current_state[name]['grad_params'] = {param_name: param.grad for param_name, param in module.named_parameters()}
        return hook
    
    def _evaluate_condition(self, condition: str, output):
        try:
            return eval(condition, {'output': output, 'torch': torch})
        except Exception as e:
            print(f"Error evaluating breakpoint condition: {str(e)}")
            return False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def get_parameter(self, param_name: str) -> Optional[nn.Parameter]:
        return dict(self.model.named_parameters()).get(param_name)

    def set_breakpoint(self, layer_name: str, condition: Optional[str] = None, action: Optional[str] = None):
        if layer_name not in self.breakpoints:
            self.breakpoints[layer_name] = []
        self.breakpoints[layer_name].append({'condition': condition, 'action': action})

    def remove_breakpoint(self, layer_name: str):
        if layer_name in self.breakpoints:
            del self.breakpoints[layer_name]

    def clear_all_breakpoints(self):
        self.breakpoints.clear()

    def reset_to_initial_state(self):
        self.model.load_state_dict(self.initial_state_dict)
        print("Model reset to initial state.")

    def step_to_layer(self, target_layer: str):
        if not self.current_state:
            return "Error: No execution state. Run the model first."
        
        def find_layer(model, target):
            for name, child in model.named_children():
                if name == target:
                    return child
                result = find_layer(child, target)
                if result is not None:
                    return result
            return None

        target_module = find_layer(self.model, target_layer)
        if target_module is None:
            return f"Error: Layer {target_layer} not found"
    
    def get_attention_weights(self, layer_name: str):
        return self.attention_weights.get(layer_name)
    
    def modify_weight(self, layer_name: str, weight_name: str, indices, value):
        layer = self.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, weight_name):
            return f"Weight '{weight_name}' not found in layer '{layer_name}'."

        weight = getattr(layer, weight_name)
        if not isinstance(weight, torch.Tensor):
            return f"'{weight_name}' is not a tensor in layer '{layer_name}'."

        try:
            with torch.no_grad():
                original_value = weight[indices].clone()
                new_weight = weight.clone()
                new_weight[indices] = value
                setattr(layer, weight_name, nn.Parameter(new_weight))
                
            self.modified_weights[(layer_name, weight_name, indices)] = original_value
            return f"Weight at {layer_name}.{weight_name}{indices} modified to {value}"
        except Exception as e:
            return f"Error modifying weight: {str(e)}"

    def reset_modified_weights(self):
        with torch.no_grad():
            for (layer_name, weight_name, indices), original_value in self.modified_weights.items():
                layer = self.get_layer(layer_name)
                weight = getattr(layer, weight_name)
                new_weight = weight.clone()
                new_weight[indices] = original_value
                setattr(layer, weight_name, nn.Parameter(new_weight))
        self.modified_weights.clear()
        return "All modified weights have been reset."
    

    def modify_activation(self, layer_name: str, function_str: str):
        layer = self.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        try:
            modification_function = eval(f"lambda x: {function_str}")
        except Exception as e:
            return f"Error in function definition: {str(e)}"

        def hook(module, input, output):
            return modification_function(output)

        # Remove existing hook if there is one
        if layer_name in self.activation_hooks:
            self.activation_hooks[layer_name].remove()

        handle = layer.register_forward_hook(hook)
        self.activation_hooks[layer_name] = handle

        return f"Activation modification hook set for layer '{layer_name}'"

    def clear_activation_modifications(self):
        for handle in self.activation_hooks.values():
            handle.remove()
        self.activation_hooks.clear()
        return "All activation modifications cleared"

    def get_layer(self, layer_name: str) -> Optional[nn.Module]:
        parts = layer_name.split('.')
        current = self.model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current



    
    

# ./core/debugger.py
import torch
from typing import Any, Dict, Optional, List, Callable
from transformers import AutoTokenizer
import struct
import cloudpickle
from contextlib import contextmanager
import traceback

from utils.error_handling import handle_exceptions
from .model_wrapper import ModelWrapper
from .execution_engine import ExecutionEngine
from inspection import ModelInspector, LayerInspector, WeightInspector, ActivationInspector, GradientInspector, AttentionInspector, VariableInspector
from breakpoints import BreakpointManager
from tracing import ExecutionTracer, ActivationTracer, GradientTracer
from analysis.token_probability import TokenProbabilityAnalyzer
from analysis.token_analyzer import TokenAnalyzer
from analysis.probe import ProbeManager, probe_decorator, ProbePoint
from analysis.dataset_example_collector import DatasetExampleCollector
from core.undo_manager import UndoManager
from advanced.custom_hooks import CustomHookManager
from experiments.experiment_manager import ExperimentManager
import socket
import pickle

class NNGDB:
    def __init__(self, model: Optional[torch.nn.Module], model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.socket = None

        if model is not None:
            self._initialize_components(model)
        else:
            self._initialize_dummy_components()

    def _initialize_components(self, model):
        self.wrapped_model = ModelWrapper(model)
        self.execution_engine = ExecutionEngine(self.wrapped_model)
        self.breakpoint_manager = BreakpointManager(self.wrapped_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_inspector = ModelInspector(self.wrapped_model)
        self.layer_inspector = LayerInspector(self.wrapped_model)
        self.weight_inspector = WeightInspector(self.wrapped_model)
        self.activation_inspector = ActivationInspector(self.wrapped_model)
        self.gradient_inspector = GradientInspector(self.wrapped_model)
        self.attention_inspector = AttentionInspector(self.wrapped_model)
        self.variable_inspector = VariableInspector(self.wrapped_model)
        self.execution_tracer = ExecutionTracer(self.wrapped_model)
        self.activation_tracer = ActivationTracer(self.wrapped_model)
        self.gradient_tracer = GradientTracer(self.wrapped_model)
        self.context: Dict[str, Any] = {}
        self.undo_manager = UndoManager()
        self.token_analyzer = TokenProbabilityAnalyzer(model, self.tokenizer)
        self.custom_hook_manager = CustomHookManager(self.wrapped_model.model)
        self.token_analyzer = TokenAnalyzer(self.wrapped_model.model, self.tokenizer, self.device)
        self.experiment_manager = ExperimentManager(self.wrapped_model.model)
        self.experiment_manager.create_experiment("base")
        self.experiment_manager.switch_experiment("base")
        self.probe_manager = ProbeManager()
        self.active_probes = []
        self._register_probe_points()
        self.dataset_example_collector = DatasetExampleCollector()

    def _initialize_dummy_components(self):
        self.wrapped_model = None
        self.execution_engine = None
        self.breakpoint_manager = None
        self.tokenizer = None
        self.model_inspector = None
        self.layer_inspector = None
        self.weight_inspector = None
        self.activation_inspector = None
        self.gradient_inspector = None
        self.attention_inspector = None
        self.variable_inspector = None
        self.execution_tracer = None
        self.activation_tracer = None
        self.gradient_tracer = None
        self.context = {}
        self.undo_manager = None
        self.token_analyzer = None
        self.custom_hook_manager = None
        self.experiment_manager = None
        self.probe_manager = None
        self.active_probes = []
        self.dataset_example_collector = None

    def _register_probe_points(self):
        if self.wrapped_model is not None:
            for name, _ in self.wrapped_model.model.named_modules():
                self.probe_manager.register_probe_point(name)

    @handle_exceptions
    def connect(self, host='localhost', port=5000):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))

    @handle_exceptions
    def execute_remote(self, method, *args, **kwargs):
        if self.socket is None:
            raise Exception("Not connected to a server. Use 'connect' first.")
        command = {
            'method': method,
            'args': args,
            'kwargs': kwargs
        }
        self.send_msg(cloudpickle.dumps(command))
        return cloudpickle.loads(self.recv_msg())

    @handle_exceptions
    def send_msg(self, msg):
        msg = struct.pack('>I', len(msg)) + msg
        self.socket.sendall(msg)

    @handle_exceptions
    def recv_msg(self):
        raw_msglen = self.recvall(4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        return self.recvall(msglen)

    @handle_exceptions
    def recvall(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    @handle_exceptions
    @probe_decorator
    def run(self, input_text: str):
        if self.socket:
            return self.execute_remote('run', input_text)
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            print(f"Input shape: {inputs.input_ids.shape}")

            with torch.no_grad():
                outputs = self.wrapped_model.model(**inputs)

            print(f"Output type: {type(outputs)}")
            if hasattr(outputs, 'logits'):
                print(f"Logits shape: {outputs.logits.shape}")
            
            if hasattr(outputs, 'logits'):
                output_ids = outputs.logits[:, -1:].argmax(dim=-1)
            elif isinstance(outputs, tuple):
                output_ids = outputs[0][:, -1:]
            else:
                output_ids = outputs[:, -1:]

            print(f"Output IDs shape: {output_ids.shape}")

            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            recordings = self.get_recordings()
            return f"Input: {input_text}\nOutput: {output_text}\nProbe recordings: {recordings}"
        except Exception as e:
            return f"Error: {str(e)}\nError type: {type(e)}"

    @handle_exceptions
    @probe_decorator
    def step(self, num_steps: int = 1):
        if self.socket:
            return self.execute_remote('step', num_steps)
        result = self.execution_engine.step(num_steps)
        recordings = self.get_recordings()
        return f"{result}\nProbe recordings: {recordings}"

    @handle_exceptions
    def set_breakpoint(self, layer_name: str, condition: str = None):
        if self.socket:
            return self.execute_remote('set_breakpoint', layer_name, condition)
        result = self.breakpoint_manager.set_breakpoint(layer_name, condition)
        self.probe_point(layer_name).probe(lambda save_ctx, tensor: setattr(save_ctx, 'breakpoint_tensor', tensor))
        return result

    @handle_exceptions
    def trace_execution(self):
        if self.socket:
            return self.execute_remote('trace_execution')
        self.execution_tracer.trace()
        for layer_name in self.execution_tracer.traced_layers:
            self.probe_point(layer_name).probe(lambda save_ctx, tensor: setattr(save_ctx, 'traced_tensor', tensor))
        return "Execution tracing enabled with probes. Run the model to collect the trace."

    @handle_exceptions
    def get_recordings(self):
        if self.socket:
            return self.execute_remote('get_recordings')
        return self.probe_manager.recordings

    @handle_exceptions
    def clear_recordings(self):
        if self.socket:
            return self.execute_remote('clear_recordings')
        self.probe_manager.clear_recordings()

    @handle_exceptions
    def probe_point(self, name: str) -> Optional[ProbePoint]:
        if self.socket:
            return self.execute_remote('probe_point', name)
        point = self.probe_manager.get_probe_point(name)
        if point is None:
            print(f"Warning: Probe point '{name}' not found. Available probe points: {', '.join(self.probe_manager.probe_points.keys())}")
        return point

    @handle_exceptions
    def add_probe(self, point_name: str, probe_function: Callable):
        if self.socket:
            return self.execute_remote('add_probe', point_name, probe_function)
        probe_point = self.probe_point(point_name)
        if probe_point is None:
            return f"Error: Probe point '{point_name}' not found. Make sure the model is initialized and the layer name is correct."
        probe_point.probe(probe_function)
        self.active_probes.append(probe_point)
        return f"Probe added to {point_name}"

    @handle_exceptions
    def clear_probes(self):
        if self.socket:
            return self.execute_remote('clear_probes')
        for probe_point in self.active_probes:
            probe_point.clear()
        self.active_probes.clear()
        return "All probes cleared"

    @handle_exceptions
    def list_probes(self):
        if self.socket:
            return self.execute_remote('list_probes')
        return {probe.name: str(probe.hooks[0]) for probe in self.active_probes}

    @handle_exceptions
    def set_context(self, key: str, value: Any):
        if self.socket:
            return self.execute_remote('set_context', key, value)
        self.context[key] = value

    @handle_exceptions
    def get_context(self, key: str) -> Any:
        if self.socket:
            return self.execute_remote('get_context', key)
        return self.context.get(key)

    @handle_exceptions
    def inspect_model(self):
        if self.socket:
            return self.execute_remote('inspect_model')
        model_info = self.model_inspector.inspect()
        return self._format_model_info(model_info)

    def _format_model_info(self, model_info):
        formatted = f"Model: {model_info['model_type']}\n"
        formatted += f"Total parameters: {model_info['num_parameters']}\n"
        formatted += f"Trainable parameters: {model_info['num_trainable_parameters']}\n"
        formatted += "Layers:\n"
        for name, layer_info in model_info['layers'].items():
            formatted += f"  {name}: {layer_info['type']}\n"
            for param_name, param_info in layer_info['parameters'].items():
                formatted += f"    {param_name}: shape={param_info['shape']}, requires_grad={param_info['requires_grad']}\n"
        return formatted

    @handle_exceptions
    def inspect_layer(self, layer_name: str):
        if self.socket:
            return self.execute_remote('inspect_layer', layer_name)
        return self.layer_inspector.inspect(layer_name)

    @handle_exceptions
    def inspect_weights(self, layer_name: str):
        if self.socket:
            return self.execute_remote('inspect_weights', layer_name)
        return self.weight_inspector.inspect(layer_name)

    @handle_exceptions
    def inspect_activations(self, layer_name: str):
        if self.socket:
            return self.execute_remote('inspect_activations', layer_name)
        return self.activation_inspector.inspect(layer_name)

    @handle_exceptions
    def inspect_gradients(self, layer_name: str):
        if self.socket:
            return self.execute_remote('inspect_gradients', layer_name)
        return self.gradient_inspector.inspect(layer_name)

    @handle_exceptions
    def inspect_attention(self, layer_name: str):
        if self.socket:
            return self.execute_remote('inspect_attention', layer_name)
        return self.attention_inspector.inspect(layer_name)

    @handle_exceptions
    def inspect_variable(self, variable_name: str):
        if self.socket:
            return self.execute_remote('inspect_variable', variable_name)
        return self.variable_inspector.inspect(variable_name)

    @handle_exceptions
    def set_breakpoint(self, layer_name: str, condition: str = None):
        if self.socket:
            return self.execute_remote('set_breakpoint', layer_name, condition)
        return self.breakpoint_manager.set_breakpoint(layer_name, condition)

    @handle_exceptions
    def remove_breakpoint(self, layer_name: str):
        if self.socket:
            return self.execute_remote('remove_breakpoint', layer_name)
        return self.breakpoint_manager.remove_breakpoint(layer_name)

    @handle_exceptions
    def list_breakpoints(self):
        if self.socket:
            return self.execute_remote('list_breakpoints')
        return self.breakpoint_manager.list_breakpoints()

    @handle_exceptions
    def trace_activations(self):
        if self.socket:
            return self.execute_remote('trace_activations')
        return self.activation_tracer.trace()

    @handle_exceptions
    def trace_gradients(self):
        if self.socket:
            return self.execute_remote('trace_gradients')
        return self.gradient_tracer.trace()

    @handle_exceptions
    def get_activation_trace(self, layer_name: str):
        if self.socket:
            return self.execute_remote('get_activation_trace', layer_name)
        return self.activation_tracer.get_activation(layer_name)
    
    @handle_exceptions
    def get_activation(self, layer_name: str, input_text: str):
        if self.socket:
            return self.execute_remote('get_activation', layer_name, input_text)
    
        def capture_activation(save_ctx, tensor):
            save_ctx.activation = tensor

        self.add_probe(layer_name, capture_activation)
        self.run(input_text)
        activation = self.get_recordings()[layer_name].activation
        self.clear_probes()
        return activation
    
    @contextmanager
    def temporary_probe(self, layer_name: str, probe_function: Callable):
        self.add_probe(layer_name, probe_function)
        try:
            yield
        finally:
            self.clear_probes()
    
    @handle_exceptions
    def get_multiple_activations(self, layer_names: List[str], input_text: str):
        if self.socket:
            return self.execute_remote('get_multiple_activations', layer_names, input_text)
    
        for layer_name in layer_names:
            self.add_probe(layer_name, lambda save_ctx, tensor: setattr(save_ctx, 'activation', tensor))
    
        self.run(input_text)
        activations = {layer: self.get_recordings()[layer].activation for layer in layer_names}
        self.clear_probes()
        return activations
    
    @handle_exceptions
    def get_execution_trace(self):
        if self.socket:
            return self.execute_remote('get_execution_trace')
        return self.execution_tracer.get_trace()
    
    @handle_exceptions
    def clear_all_traces(self):
        if self.socket:
            return self.execute_remote('clear_all_traces')
        self.execution_tracer.clear_trace()
        self.activation_tracer.clear_trace()
        self.gradient_tracer.clear_trace()
        return "All traces cleared."
    
    @handle_exceptions
    def continue_execution(self):
        if self.socket:
            return self.execute_remote('continue_execution')
        return self.execution_engine.continue_execution()
    
    @handle_exceptions
    def get_gradient_trace(self, layer_name: str):
        if self.socket:
            return self.execute_remote('get_gradient_trace', layer_name)
        return self.gradient_tracer.get_gradient(layer_name)
    
    @handle_exceptions
    def modify_weight(self, layer_name: str, weight_name: str, indices, value):
        if self.socket:
            return self.execute_remote('modify_weight', layer_name, weight_name, indices, value)
        return self.wrapped_model.modify_weight(layer_name, weight_name, indices, value)

    @handle_exceptions
    def modify_activation(self, layer_name: str, function_str: str):
        if self.socket:
            return self.execute_remote('modify_activation', layer_name, function_str)
        try:
            result = self.wrapped_model.modify_activation(layer_name, function_str)
            return result
        except Exception as e:
            return f"Error modifying activation: {str(e)}"

    @handle_exceptions
    def reset_modified_weights(self):
        if self.socket:
            return self.execute_remote('reset_modified_weights')
        return self.wrapped_model.reset_modified_weights()

    @handle_exceptions  
    def get_layer_representation(self, layer_name: str):
        if self.socket:
            return self.execute_remote('get_layer_representation', layer_name)
        if layer_name not in self.wrapped_model.current_state:
            return f"No data available for layer '{layer_name}'."
        return self.wrapped_model.current_state[layer_name]['output']
    
    @handle_exceptions
    def add_hook(self, hook_type: str, module_name: str, hook_name: str, hook_function: str):
        if self.socket:
            return self.execute_remote('add_hook', hook_type, module_name, hook_name, hook_function)
        if hook_type == "forward":
            return self.custom_hook_manager.register_forward_hook(module_name, eval(hook_function), hook_name)
        elif hook_type == "backward":
            return self.custom_hook_manager.register_backward_hook(module_name, eval(hook_function), hook_name)
        else:
            return "Invalid hook type. Use 'forward' or 'backward'."
    
    @handle_exceptions
    def remove_hook(self, hook_name: str):
        if self.socket:
            return self.execute_remote('remove_hook', hook_name)
        return self.custom_hook_manager.remove_hook(hook_name)
    
    @handle_exceptions
    def list_hooks(self):
        if self.socket:
            return self.execute_remote('list_hooks')
        return self.custom_hook_manager.list_hooks()

    @handle_exceptions
    def clear_hooks(self):
        if self.socket:
            return self.execute_remote('clear_hooks')
        return self.custom_hook_manager.clear_all_hooks()
    
    @handle_exceptions
    def analyze_tokens(self, input_text: str, analysis_type: str, compare_modified: bool = False, **kwargs):
        if self.socket:
            return self.execute_remote('analyze_tokens', input_text, analysis_type, compare_modified, **kwargs)
        result = self.token_analyzer.analyze(input_text, analysis_type, compare_modified, **kwargs)
        
        if analysis_type == 'probabilities' and compare_modified:
            comparison = f"Original analysis:\n{self._format_token_analysis(result['original'])}\n\n"
            comparison += f"Analysis with modified weights:\n{self._format_token_analysis(result['modified'])}"
            return comparison
        elif analysis_type == 'probabilities':
            return self._format_token_analysis(result)
        elif analysis_type == 'saliency':
            return self._format_saliency_analysis(result)
        elif analysis_type == 'attention':
            return self._format_attention_analysis(result)
        else:
            return result
    
    @handle_exceptions
    def analyze_token_attention_and_representation(self, input_text: str, **kwargs):
        if self.socket:
            return self.execute_remote('analyze_token_attention_and_representation', input_text, **kwargs)
        return self.token_analyzer.analyze_attention_and_representation(input_text, **kwargs)

    @handle_exceptions
    def undo(self):
        if self.socket:
            return self.execute_remote('undo')
        state = self.undo_manager.undo()
        if state:
            self.wrapped_model.load_state_dict(state)
            return "Undo successful"
        return "Nothing to undo"

    @handle_exceptions
    def redo(self):
        if self.socket:
            return self.execute_remote('redo')
        state = self.undo_manager.redo()
        if state:
            self.wrapped_model.load_state_dict(state)
            return "Redo successful"
        return "Nothing to redo"

    @handle_exceptions
    def create_experiment(self, name):
        if self.socket:
            return self.execute_remote('create_experiment', name)
        return self.experiment_manager.create_experiment(name)

    @handle_exceptions
    def switch_experiment(self, name):
        if self.socket:
            return self.execute_remote('switch_experiment', name)
        return self.experiment_manager.switch_experiment(name)

    @handle_exceptions
    def list_experiments(self):
        if self.socket:
            return self.execute_remote('list_experiments')
        return self.experiment_manager.list_experiments()

    @handle_exceptions
    def delete_experiment(self, name):
        if self.socket:
            return self.execute_remote('delete_experiment', name)
        return self.experiment_manager.delete_experiment(name)

    @handle_exceptions
    def get_current_experiment(self):
        if self.socket:
            return self.execute_remote('get_current_experiment')
        return self.experiment_manager.get_current_experiment()

    @handle_exceptions
    def compare_experiments(self, exp1, exp2, input_text, analysis_type='probabilities', **kwargs):
        if self.socket:
            return self.execute_remote('compare_experiments', exp1, exp2, input_text, analysis_type, **kwargs)
        current_exp = self.get_current_experiment()
        
        self.switch_experiment(exp1)
        result1 = self.analyze_tokens(input_text, analysis_type, **kwargs)
        
        self.switch_experiment(exp2)
        result2 = self.analyze_tokens(input_text, analysis_type, **kwargs)
        
        # Switch back to the original experiment
        if current_exp:
            self.switch_experiment(current_exp)
        
        comparison = f"Experiment '{exp1}':\n{result1}\n\n"
        comparison += f"Experiment '{exp2}':\n{result2}"
        return comparison

    @handle_exceptions
    def _format_token_analysis(self, analysis):
        formatted = f"Input: {analysis['input_text']}\nTop {len(analysis['top_tokens'])} tokens:\n"
        formatted += "\n".join([f"{token}: {prob:.4f}" for token, prob in analysis['top_tokens']])
        return formatted
    
    @handle_exceptions
    def _format_saliency_analysis(self, analysis):
        formatted = "Saliency Analysis:\n"
        for token, saliency in zip(analysis['tokens'], analysis['saliency']):
            formatted += f"{token}: {saliency:.4f}\n"
        return formatted
    
    def _format_attention_analysis(self, analysis):
        formatted = "Attention Analysis:\n"
        formatted += f"Tokens: {', '.join(analysis['tokens'])}\n"
        formatted += "Attention weights:\n"
        for i, token in enumerate(analysis['tokens']):
            formatted += f"{token}: {analysis['attention_weights'][i]}\n"
        return formatted
    
    def __getattr__(self, name):
        if self.socket:
            return lambda *args, **kwargs: self.execute_remote(name, *args, **kwargs)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    @handle_exceptions
    def collect_dataset_examples(self, input_texts: List[str], layer_names: List[str], top_n: int = 10):
        if self.socket:
            return self.execute_remote('collect_dataset_examples', input_texts, layer_names, top_n)

        self.dataset_example_collector.clear()
        self.dataset_example_collector.num_top_examples = top_n

        def activation_hook(module, input, output):
            try:
                layer_name = next(name for name, mod in self.wrapped_model.model.named_modules() if mod is module)
                if layer_name in layer_names:
                    # Assuming input is a tuple and the first element is the input_ids
                    input_ids = input[0] if isinstance(input, tuple) else input
                    if isinstance(input_ids, torch.Tensor):
                        input_ids = input_ids.squeeze().tolist()
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                    if isinstance(tokens, str):
                        tokens = [tokens]  # Ensure tokens is always a list
                    print(f"Debug - Layer: {layer_name}, Tokens: {tokens}, Output shape: {output.shape}")
                    self.dataset_example_collector.collect_activations(layer_name, output[0], tokens)
            except Exception as e:
                print(f"Error in activation_hook: {str(e)}")
                print(traceback.format_exc())

# ./utils/data_generator.py
import torch

def generate_random_input(input_shape: tuple, device: str = 'cpu'):
    return torch.randn(input_shape).to(device)

def generate_adversarial_input(model: torch.nn.Module, original_input: torch.Tensor, target_class: int, epsilon: float = 0.01, num_steps: int = 10):
    perturbed_input = original_input.clone().detach().requires_grad_(True)
    
    for _ in range(num_steps):
        output = model(perturbed_input)
        loss = -output[0, target_class]
        loss.backward()
        
        with torch.no_grad():
            perturbed_input += epsilon * perturbed_input.grad.sign()
            perturbed_input.clamp_(0, 1)  # Assuming input values are between 0 and 1
        
        perturbed_input.grad.zero_()
    
    return perturbed_input.detach()

# ./utils/__init.py
from .tensor_utils import *
from .data_generator import *
from .performance_utils import *

__all__ = [
    'tensor_stats',
    'tensor_histogram',
    'generate_random_input',
    'generate_adversarial_input',
    'measure_inference_time',
    'profile_memory_usage'
]

# ./utils/tensor_utils.py
import torch

def tensor_stats(tensor: torch.Tensor):
    return {
        "shape": tensor.shape,
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "norm": tensor.norm().item(),
    }

def tensor_histogram(tensor: torch.Tensor, num_bins: int = 10):
    hist = torch.histogram(tensor.float().view(-1), bins=num_bins)
    return {
        "bin_edges": hist.bin_edges.tolist(),
        "counts": hist.count.tolist()
    }

# ./utils/logger.py
# nngdb/utils/logger.py

import logging
import sys

def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(f'{name}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# ./utils/performance_utils.py
import time
import torch

def measure_inference_time(model: torch.nn.Module, input_tensor: torch.Tensor, num_runs: int = 100):
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    return avg_time

def profile_memory_usage(model: torch.nn.Module, input_tensor: torch.Tensor):
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    output = model(input_tensor)
    
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_usage = final_memory - initial_memory
    
    return memory_usage

# ./utils/error_handling.py
class NNGDBException(Exception):
    pass

def handle_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NNGDBException as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    return wrapper

# ./utils/activation_utils.py
import torch

def analyze_activation(activation: torch.Tensor):
    return {
        "shape": activation.shape,
        "mean": activation.mean().item(),
        "std": activation.std().item(),
        "min": activation.min().item(),
        "max": activation.max().item(),
        "fraction_zeros": (activation == 0).float().mean().item(),
    }

# ./advanced/__init__.py
from .custom_hooks import CustomHookManager


__all__ = ['CustomHookManager']

# ./advanced/custom_hooks.py
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

# ./tracing/__init__.py
from .execution_tracer import ExecutionTracer
from .activation_tracer import ActivationTracer
from .gradient_tracer import GradientTracer

__all__ = ['ExecutionTracer', 'ActivationTracer', 'GradientTracer']

# ./tracing/execution_tracer.py
from core.model_wrapper import ModelWrapper

class ExecutionTracer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.execution_trace = []

    def trace(self):
        self.execution_trace = []
        self._register_hooks()
        return "Execution tracing enabled. Run the model to collect the trace."

    def _register_hooks(self):
        def hook(module, input, output):
            self.execution_trace.append({
                'layer_name': module.__class__.__name__,
                'input_shape': [tuple(i.shape) for i in input if hasattr(i, 'shape')],
                'output_shape': tuple(output.shape) if hasattr(output, 'shape') else None
            })

        for name, module in self.wrapped_model.model.named_modules():
            module.register_forward_hook(hook)

    def get_trace(self):
        return self.execution_trace

    def clear_trace(self):
        self.execution_trace = []
        return "Execution trace cleared."

    def summarize_trace(self):
        summary = []
        for idx, step in enumerate(self.execution_trace):
            summary.append(f"Step {idx}: {step['layer_name']} - Input: {step['input_shape']}, Output: {step['output_shape']}")
        return "\n".join(summary)

# ./tracing/gradient_tracer.py
import torch
from core.model_wrapper import ModelWrapper

class GradientTracer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.gradient_trace = {}

    def trace(self):
        self.gradient_trace = {}
        self._register_hooks()
        return "Gradient tracing enabled. Run backward pass to collect gradients."

    def _register_hooks(self):
        def hook(module, grad_input, grad_output):
            self.gradient_trace[module.__class__.__name__] = {
                'grad_input': [g.detach() if isinstance(g, torch.Tensor) else g for g in grad_input],
                'grad_output': [g.detach() if isinstance(g, torch.Tensor) else g for g in grad_output]
            }

        for name, module in self.wrapped_model.model.named_modules():
            module.register_full_backward_hook(hook)

    def get_gradient(self, layer_name: str):
        return self.gradient_trace.get(layer_name, f"No gradient recorded for layer '{layer_name}'")

    def get_all_gradients(self):
        return self.gradient_trace

    def clear_trace(self):
        self.gradient_trace = {}
        return "Gradient trace cleared."

    def summarize_trace(self):
        summary = []
        for layer_name, grads in self.gradient_trace.items():
            summary.append(f"Layer: {layer_name}")
            for grad_type in ['grad_input', 'grad_output']:
                for idx, grad in enumerate(grads[grad_type]):
                    if isinstance(grad, torch.Tensor):
                        summary.append(f"  {grad_type}[{idx}]: shape={grad.shape}, mean={grad.mean().item():.4f}, std={grad.std().item():.4f}")
                    else:
                        summary.append(f"  {grad_type}[{idx}]: {type(grad)}")
        return "\n".join(summary)

# ./tracing/activation_tracer.py
import torch
from core.model_wrapper import ModelWrapper

class ActivationTracer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.activation_trace = {}

    def trace(self):
        def hook(module, input, output):
            self.activation_trace[module.__class__.__name__] = output.detach()

        for name, module in self.wrapped_model.model.named_modules():
            module.register_forward_hook(hook)

    def _register_hooks(self):
        def hook(module, input, output):
            self.activation_trace[module.__class__.__name__] = output.detach() if isinstance(output, torch.Tensor) else output

        for name, module in self.wrapped_model.model.named_modules():
            module.register_forward_hook(hook)

    def get_activation(self, layer_name: str):
        return self.activation_trace.get(layer_name, f"No activation recorded for layer '{layer_name}'")

    def get_all_activations(self):
        return self.activation_trace

    def clear_trace(self):
        self.activation_trace = {}
        return "Activation trace cleared."

    def summarize_trace(self):
        summary = []
        for layer_name, activation in self.activation_trace.items():
            if isinstance(activation, torch.Tensor):
                summary.append(f"{layer_name}: shape={activation.shape}, mean={activation.mean().item():.4f}, std={activation.std().item():.4f}")
            else:
                summary.append(f"{layer_name}: {type(activation)}")
        return "\n".join(summary)

# ./analysis/__init__.py
from .gradient_flow import GradientFlowAnalyzer
from .attention_analysis import AttentionAnalyzer
from .neuron_activation import NeuronActivationAnalyzer
from .perturbation_analysis import PerturbationAnalyzer
from .token_probability import TokenProbabilityAnalyzer
from .token_analyzer import TokenAnalyzer
from .probe import ProbeManager, ProbePoint, probe_decorator
from .dataset_example_collector import DatasetExampleCollector

__all__ = [
    'GradientFlowAnalyzer',
    'AttentionAnalyzer',
    'NeuronActivationAnalyzer',
    'PerturbationAnalyzer',
    'TokenProbabilityAnalyzer',
    'TokenAnalyzer',
    'ProbeManager',
    'ProbePoint',
    'probe_decorator'
    'DatasetExampleCollector'
]

# ./analysis/token_analyzer.py
# analysis/token_analyzer.py

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from transformers import AutoModel, AutoModelForCausalLM

#Experimental
class TokenAnalyzer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.history = []

    def analyze(self, input_text: str, analysis_type: str, compare_modified: bool = False, **kwargs):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        if analysis_type == 'probabilities':
            original_result = self._analyze_token_probabilities(input_ids, tokens, **kwargs)
            if not compare_modified:
                return original_result
            
            # If comparing with modified weights, perform the analysis again
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            probs = torch.softmax(logits[0, -1], dim=-1)
            top_k = kwargs.get('top_k', 5)
            top_probs, top_indices = torch.topk(probs, top_k)
            
            modified_result = {
                "input_text": input_text,
                "top_tokens": [
                    (self.tokenizer.decode([idx.item()]), prob.item())
                    for idx, prob in zip(top_indices, top_probs)
                ]
            }
            
            return {
                "original": original_result,
                "modified": modified_result
            }
        else:
            analysis_methods = {
                'saliency': self._token_saliency,
                'attention': self._visualize_attention,
                'counterfactual': self._counterfactual_analysis,
                'attribution': self._token_attribution,
                'neuron_activation': self._neuron_activation_by_token,
                'representation_tracking': self._track_token_representations,
                'clustering': self._cluster_tokens,
                'importance_ranking': self._rank_token_importance
            }

            if analysis_type not in analysis_methods:
                return f"Unknown analysis type: {analysis_type}"

            return analysis_methods[analysis_type](input_ids, tokens, **kwargs)

    def analyze_attention_and_representation(self, input_text: str, layer: int = -1, head: int = None, 
                                             include_attention: bool = True, include_representation: bool = True) -> Dict[str, Any]:
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        results = {}
        
        if include_attention:
            results['attention'] = self._visualize_attention(input_ids, tokens, layer=layer, head=head)
        
        if include_representation:
            results['representation'] = self._track_token_representations(input_ids, tokens)
        
        return results

    def _analyze_token_probabilities(self, input_ids: torch.Tensor, tokens: List[str], top_k: int = 5) -> Dict[str, Any]:
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        probs = F.softmax(logits[0, -1], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        return {
            "input_text": self.tokenizer.decode(input_ids[0]),
            "top_tokens": [
                (self.tokenizer.decode([idx.item()]), prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
        }

    def _token_saliency(self, input_ids, tokens):
        self.model.zero_grad()
        embed = self.model.get_input_embeddings()
        
        input_ids = input_ids.to(self.model.device)
        input_embed = embed(input_ids)
        input_embed.retain_grad()
        
        outputs = self.model(inputs_embeds=input_embed)
        output = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        output.sum().backward()
        
        saliency = input_embed.grad.abs().sum(dim=-1)
        
        return {
            "tokens": tokens,
            "saliency": saliency[0].tolist()
        }

    def _visualize_attention(self, input_ids, tokens, layer=None, head=None):
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Determine the number of layers
        if isinstance(self.model, AutoModelForCausalLM):
            num_layers = len(self.model.transformer.h)
        elif hasattr(self.model, 'config'):
            num_layers = self.model.config.num_hidden_layers
        else:
            return "Unable to determine the number of layers in the model."

        # Handle layer selection
        layer = num_layers - 1 if layer is None else int(layer)
        layer = layer if layer >= 0 else num_layers + layer

        if layer < 0 or layer >= num_layers:
            return f"Invalid layer index. Model has {num_layers} layers."

        # Prepare inputs
        input_ids = input_ids.to(self.model.device)

        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)

        # Extract attention weights
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions[layer]
        else:
            return "Model does not output attention weights. Make sure you're using a model that supports outputting attention."

        # Handle head selection
        if head is not None:
            head = int(head)
            if head < 0 or head >= attentions.size(1):
                return f"Invalid head index. Layer has {attentions.size(1)} attention heads."
            attentions = attentions[:, head, :, :]
        else:
            attentions = attentions.mean(dim=1)

        attention_data = attentions[0].cpu().numpy()

        return {
            "tokens": tokens,
            "attention_weights": attention_data.tolist()
        }

    def _counterfactual_analysis(self, input_ids: torch.Tensor, tokens: List[str], token_index: int, replacement_tokens: List[str]) -> Dict[str, Any]:
        original_output = self.model(input_ids).logits
        results = []
        
        for replacement in replacement_tokens:
            replacement_id = self.tokenizer.encode(replacement, add_special_tokens=False)[0]
            new_input_ids = input_ids.clone()
            new_input_ids[0, token_index] = replacement_id
            new_output = self.model(new_input_ids).logits
            
            diff = (new_output - original_output).abs().mean().item()
            results.append((replacement, diff))
        
        return {
            "original_token": tokens[token_index],
            "counterfactuals": results
        }

    def _token_attribution(self, input_ids: torch.Tensor, tokens: List[str], method: str = 'integrated_gradients') -> Dict[str, Any]:
        if method == 'integrated_gradients':
            attributions = self._integrated_gradients(input_ids)
        else:
            return {"error": f"Unsupported attribution method: {method}"}
        
        return {
            "tokens": tokens,
            "attributions": attributions[0].tolist()
        }

    def _integrated_gradients(self, input_ids: torch.Tensor, steps: int = 50) -> torch.Tensor:
        baseline = torch.zeros_like(input_ids)
        attributions = torch.zeros_like(input_ids, dtype=torch.float)
        
        for step in range(1, steps + 1):
            interpolated_input = baseline + (step / steps) * (input_ids - baseline)
            interpolated_input.requires_grad_(True)
            
            outputs = self.model(interpolated_input)
            output = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            output.backward(torch.ones_like(output))
            
            attributions += interpolated_input.grad
        
        attributions /= steps
        return attributions * (input_ids - baseline)

    def _neuron_activation_by_token(self, input_ids: torch.Tensor, tokens: List[str], layer: int) -> Dict[str, Any]:
        def hook_fn(module, input, output):
            self.layer_output = output.detach()
        
        layer_module = list(self.model.modules())[layer]
        handle = layer_module.register_forward_hook(hook_fn)
        
        self.model(input_ids)
        handle.remove()
        
        activations = self.layer_output[0]
        top_neurons = activations.max(dim=-1).indices
        
        return {
            "tokens": tokens,
            "top_neurons": top_neurons.tolist()
        }

    def _track_token_representations(self, input_ids: torch.Tensor, tokens: List[str]) -> Dict[str, Any]:
        representations = []
        
        def hook_fn(module, input, output):
            representations.append(output[0].detach())
        
        handles = []
        for layer in self.model.encoder.layer:
            handles.append(layer.register_forward_hook(hook_fn))
        
        self.model(input_ids)
        
        for handle in handles:
            handle.remove()
        
        return {
            "tokens": tokens,
            "representations": [rep.cpu().numpy().tolist() for rep in representations]
        }

    def _cluster_tokens(self, input_ids: torch.Tensor, tokens: List[str], layer: int, n_clusters: int = 5) -> Dict[str, Any]:
        def hook_fn(module, input, output):
            self.layer_output = output[0].detach()
        
        layer_module = list(self.model.modules())[layer]
        handle = layer_module.register_forward_hook(hook_fn)
        
        self.model(input_ids)
        handle.remove()
        
        token_embeddings = self.layer_output.squeeze(0).cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(token_embeddings)
        
        return {
            "tokens": tokens,
            "clusters": clusters.tolist()
        }

    def _rank_token_importance(self, input_ids: torch.Tensor, tokens: List[str]) -> Dict[str, Any]:
        input_ids.requires_grad_(True)
        outputs = self.model(input_ids)
        output = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        output.backward(torch.ones_like(output))
        
        importance = input_ids.grad.abs().sum(dim=-1)
        sorted_indices = importance[0].argsort(descending=True)
        
        return {
            "tokens": tokens,
            "importance_ranking": sorted_indices.tolist()
        }

# ./analysis/perturbation_analysis.py
import torch
from core.model_wrapper import ModelWrapper

class PerturbationAnalyzer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def analyze_input_perturbation(self, input_tensor: torch.Tensor, perturbation_scale: float = 0.1, num_samples: int = 10):
        original_output = self.wrapped_model.model(input_tensor)
        
        perturbed_outputs = []
        for _ in range(num_samples):
            perturbation = torch.randn_like(input_tensor) * perturbation_scale
            perturbed_input = input_tensor + perturbation
            perturbed_output = self.wrapped_model.model(perturbed_input)
            perturbed_outputs.append(perturbed_output)

        output_diff = torch.stack([output - original_output for output in perturbed_outputs])
        
        analysis = {
            "mean_output_diff": output_diff.abs().mean().item(),
            "std_output_diff": output_diff.std().item(),
            "max_output_diff": output_diff.abs().max().item(),
        }

        return analysis

    def analyze_weight_perturbation(self, layer_name: str, perturbation_scale: float = 0.1, num_samples: int = 10):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        original_weights = {name: param.clone() for name, param in layer.named_parameters() if 'weight' in name}
        original_output = self.wrapped_model.model(self.wrapped_model.current_state['input'])

        perturbed_outputs = []
        for _ in range(num_samples):
            with torch.no_grad():
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        perturbation = torch.randn_like(param) * perturbation_scale
                        param.add_(perturbation)
                
                perturbed_output = self.wrapped_model.model(self.wrapped_model.current_state['input'])
                perturbed_outputs.append(perturbed_output)

                # Restore original weights
                for name, original_weight in original_weights.items():
                    getattr(layer, name).copy_(original_weight)

        output_diff = torch.stack([output - original_output for output in perturbed_outputs])

        analysis = {
            "mean_output_diff": output_diff.abs().mean().item(),
            "std_output_diff": output_diff.std().item(),
            "max_output_diff": output_diff.abs().max().item(),
        }

        return analysis

    def compute_input_gradient(self, input_tensor: torch.Tensor):
        input_tensor.requires_grad_(True)
        output = self.wrapped_model.model(input_tensor)
        
        if isinstance(output, torch.Tensor):
            loss = output.sum()
        elif isinstance(output, tuple):
            loss = output[0].sum()  # Assuming the first element is the main output
        else:
            return "Unexpected output type. Unable to compute input gradient."

        loss.backward()

        input_gradient = input_tensor.grad

        return {
            "input_gradient_norm": input_gradient.norm().item(),
            "input_gradient_mean": input_gradient.abs().mean().item(),
            "input_gradient_std": input_gradient.std().item(),
        }

    def compute_saliency_map(self, input_tensor: torch.Tensor):
        input_tensor.requires_grad_(True)
        output = self.wrapped_model.model(input_tensor)
        
        if isinstance(output, torch.Tensor):
            loss = output.sum()
        elif isinstance(output, tuple):
            loss = output[0].sum()
        else:
            return "Unexpected output type. Unable to compute saliency map."

        loss.backward()

        saliency_map = input_tensor.grad.abs()
        
        return {
            "saliency_map": saliency_map,
            "max_saliency": saliency_map.max().item(),
            "mean_saliency": saliency_map.mean().item(),
        }

# ./analysis/probe.py
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

# ./analysis/neuron_activation.py
import torch
from core.model_wrapper import ModelWrapper

class NeuronActivationAnalyzer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def analyze(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No activation data available for layer '{layer_name}'."

        activation = self.wrapped_model.current_state[layer_name]['output']
        if not isinstance(activation, torch.Tensor):
            return f"Activation for layer '{layer_name}' is not a tensor."

        return self._analyze_activation(activation)

    def _analyze_activation(self, activation: torch.Tensor):
        analysis = {
            "shape": activation.shape,
            "mean": activation.mean().item(),
            "std": activation.std().item(),
            "min": activation.min().item(),
            "max": activation.max().item(),
            "fraction_zeros": (activation == 0).float().mean().item(),
            "top_k_active": self._get_top_k_active(activation, k=10),
            "activation_statistics": self._compute_activation_statistics(activation),
        }
        return analysis

    def _get_top_k_active(self, activation: torch.Tensor, k: int):
        if activation.dim() > 2:
            activation = activation.view(activation.size(0), -1)
        top_k_values, top_k_indices = torch.topk(activation.abs().mean(dim=0), k)
        return [(idx.item(), val.item()) for idx, val in zip(top_k_indices, top_k_values)]

    def _compute_activation_statistics(self, activation: torch.Tensor):
        if activation.dim() > 2:
            activation = activation.view(activation.size(0), -1)
        
        positive_fraction = (activation > 0).float().mean(dim=0)
        negative_fraction = (activation < 0).float().mean(dim=0)
        zero_fraction = (activation == 0).float().mean(dim=0)

        return {
            "positive_fraction": positive_fraction.mean().item(),
            "negative_fraction": negative_fraction.mean().item(),
            "zero_fraction": zero_fraction.mean().item(),
        }

    def get_most_active_neurons(self, layer_name: str, k: int = 10):
        if layer_name not in self.wrapped_model.current_state:
            return f"No activation data available for layer '{layer_name}'."

        activation = self.wrapped_model.current_state[layer_name]['output']
        if not isinstance(activation, torch.Tensor):
            return f"Activation for layer '{layer_name}' is not a tensor."

        if activation.dim() > 2:
            activation = activation.view(activation.size(0), -1)

        mean_activation = activation.abs().mean(dim=0)
        top_k_values, top_k_indices = torch.topk(mean_activation, k)

        return [(idx.item(), val.item()) for idx, val in zip(top_k_indices, top_k_values)]

    def compute_activation_similarity(self, layer_name: str, reference_input: torch.Tensor):
        if layer_name not in self.wrapped_model.current_state:
            return f"No activation data available for layer '{layer_name}'."

        current_activation = self.wrapped_model.current_state[layer_name]['output']
        if not isinstance(current_activation, torch.Tensor):
            return f"Activation for layer '{layer_name}' is not a tensor."

        with torch.no_grad():
            reference_activation = self.wrapped_model.model(reference_input)
            reference_activation = self.wrapped_model.current_state[layer_name]['output']

        similarity = torch.nn.functional.cosine_similarity(current_activation.view(-1), reference_activation.view(-1), dim=0)

        return similarity.item()

# ./analysis/dataset_example_collector.py
# nngdb/analysis/dataset_example_collector.py

import torch
import heapq
from typing import Dict, List, Tuple
import traceback

class DatasetExampleCollector:
    def __init__(self, num_top_examples: int = 10):
        self.num_top_examples = num_top_examples
        self.layer_activations: Dict[str, List[List[Tuple[float, str]]]] = {}

    def collect_activations(self, layer_name: str, activations: torch.Tensor, input_tokens: List[str]):
        if layer_name not in self.layer_activations:
            self.layer_activations[layer_name] = [[] for _ in range(activations.size(-1))]

        for neuron_idx in range(activations.size(-1)):
            neuron_activations = activations[:, neuron_idx]
            top_activations = self.layer_activations[layer_name][neuron_idx]

            for token_idx, activation in enumerate(neuron_activations):
                if token_idx < len(input_tokens):  # Ensure we don't go out of bounds
                    activation_value = activation.item()
                    token = input_tokens[token_idx]

                    if len(top_activations) < self.num_top_examples:
                        heapq.heappush(top_activations, (activation_value, token))
                    elif activation_value > top_activations[0][0]:
                        heapq.heapreplace(top_activations, (activation_value, token))

    def get_top_examples(self, layer_name: str) -> List[List[Tuple[float, str]]]:
        if layer_name not in self.layer_activations:
            return []
        return [sorted(neuron_activations, reverse=True) for neuron_activations in self.layer_activations[layer_name]]

    def clear(self):
        self.layer_activations.clear()

# ./analysis/token_probability.py
import torch

class TokenProbabilityAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.history = []

    def analyze(self, input_text, top_k=5):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        probs = torch.softmax(logits[0, -1], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        result = {
            "input_text": input_text,
            "top_tokens": [
                (self.tokenizer.decode([idx.item()]), prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
        }
        self.history.append(result)
        return result

    def compare(self, index1, index2):
        if index1 >= len(self.history) or index2 >= len(self.history):
            return f"Invalid indices for comparison. Available range: 0-{len(self.history)-1}"
        
        result1 = self.history[index1]
        result2 = self.history[index2]
        
        comparison = f"Comparison:\n"
        comparison += f"Input 1: {result1['input_text']}\n"
        comparison += f"Input 2: {result2['input_text']}\n\n"
        comparison += "Top tokens:\n"
        
        for (token1, prob1), (token2, prob2) in zip(result1['top_tokens'], result2['top_tokens']):
            comparison += f"{token1} ({prob1:.4f}) vs {token2} ({prob2:.4f})\n"
        
        return comparison

# ./analysis/gradient_flow.py
import torch
from core.model_wrapper import ModelWrapper

class GradientFlowAnalyzer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def analyze(self):
        gradients = {}
        for name, param in self.wrapped_model.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients[name] = {
                    'mean': param.grad.abs().mean().item(),
                    'std': param.grad.std().item(),
                    'max': param.grad.abs().max().item(),
                    'norm': param.grad.norm().item()
                }

        return gradients

    def detect_vanishing_exploding_gradients(self, threshold=1e-4):
        vanishing = []
        exploding = []
        for name, param in self.wrapped_model.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm < threshold:
                    vanishing.append((name, grad_norm))
                elif grad_norm > 1/threshold:
                    exploding.append((name, grad_norm))
        
        return {
            'vanishing': vanishing,
            'exploding': exploding
        }

    def compute_layer_gradients(self):
        layer_gradients = {}
        for name, module in self.wrapped_model.model.named_modules():
            if list(module.children()):  # skip container modules
                continue
            layer_grad = torch.cat([p.grad.view(-1) for p in module.parameters() if p.grad is not None])
            layer_gradients[name] = {
                'mean': layer_grad.abs().mean().item(),
                'std': layer_grad.std().item(),
                'max': layer_grad.abs().max().item(),
                'norm': layer_grad.norm().item()
            }
        return layer_gradients

# ./analysis/attention_analysis.py
import torch
from core.model_wrapper import ModelWrapper

class AttentionAnalyzer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def analyze_attention_patterns(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No attention data available for layer '{layer_name}'."

        layer_output = self.wrapped_model.current_state[layer_name]['output']
        
        if isinstance(layer_output, tuple):
            attention_weights = layer_output[1]
        elif isinstance(layer_output, torch.Tensor):
            attention_weights = layer_output
        else:
            return f"Unexpected output type for layer '{layer_name}': {type(layer_output)}"

        return self._analyze_attention(attention_weights)

    def _analyze_attention(self, attention_weights: torch.Tensor):
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        avg_attention = attention_weights.mean(dim=(0, 1))  # Average over batch and heads
        
        analysis = {
            "shape": attention_weights.shape,
            "num_heads": num_heads,
            "sequence_length": seq_len,
            "entropy": self._compute_attention_entropy(attention_weights),
            "top_k_attention": self._get_top_k_attention(avg_attention, k=5),
            "attention_to_self": self._compute_attention_to_self(avg_attention),
            "attention_to_neighbors": self._compute_attention_to_neighbors(avg_attention),
        }

        return analysis

    def _compute_attention_entropy(self, attention_weights: torch.Tensor):
        attention_probs = attention_weights.mean(dim=1)  # Average over heads
        entropy = -(attention_probs * torch.log(attention_probs + 1e-9)).sum(dim=-1).mean().item()
        return entropy

    def _get_top_k_attention(self, avg_attention: torch.Tensor, k: int):
        top_k_values, top_k_indices = torch.topk(avg_attention.mean(dim=0), k)
        return [(idx.item(), val.item()) for idx, val in zip(top_k_indices, top_k_values)]

    def _compute_attention_to_self(self, avg_attention: torch.Tensor):
        return torch.diag(avg_attention).mean().item()

    def _compute_attention_to_neighbors(self, avg_attention: torch.Tensor):
        seq_len = avg_attention.shape[0]
        neighbor_attention = torch.diag(avg_attention, diagonal=1) + torch.diag(avg_attention, diagonal=-1)
        return neighbor_attention.sum().item() / (2 * (seq_len - 1))

    def visualize_attention(self, layer_name: str):
        # This method would typically create a visualization of the attention weights.
        # For now, we'll just return a message indicating that visualization is not implemented.
        return "Attention visualization not implemented in this version."

# ./inspection/__init__.py
from .model_inspector import ModelInspector
from .layer_inspector import LayerInspector
from .weight_inspector import WeightInspector
from .activation_inspector import ActivationInspector
from .gradient_inspector import GradientInspector
from .attention_inspector import AttentionInspector
from .variable_inspector import VariableInspector

__all__ = [
    'ModelInspector',
    'LayerInspector',
    'WeightInspector',
    'ActivationInspector',
    'GradientInspector',
    'AttentionInspector',
    'VariableInspector'
]

# ./inspection/weight_inspector.py
import torch
from core.model_wrapper import ModelWrapper

class WeightInspector:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def inspect(self, layer_name: str):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        weight_info = {}
        for name, param in layer.named_parameters():
            if 'weight' in name:
                weight_info[name] = self._analyze_weight(param)
        return weight_info

    def _analyze_weight(self, weight: torch.Tensor):
        return {
            "shape": weight.shape,
            "mean": weight.mean().item(),
            "std": weight.std().item(),
            "min": weight.min().item(),
            "max": weight.max().item(),
            "norm": weight.norm().item(),
            "num_zeros": (weight == 0).sum().item(),
            "num_non_zeros": (weight != 0).sum().item(),
        }

    def get_weight(self, layer_name: str, weight_name: str):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."
        
        for name, param in layer.named_parameters():
            if name == weight_name:
                return param
        return f"Weight '{weight_name}' not found in layer '{layer_name}'."

# ./inspection/activation_inspector.py
import torch
from core.model_wrapper import ModelWrapper

class ActivationInspector:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def inspect(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No activation data available for layer '{layer_name}'."

        activation = self.wrapped_model.current_state[layer_name]['output']
        if not isinstance(activation, torch.Tensor):
            return f"Activation for layer '{layer_name}' is not a tensor."

        return self._analyze_activation(activation)

    def _analyze_activation(self, activation: torch.Tensor):
        return {
            "shape": activation.shape,
            "mean": activation.mean().item(),
            "std": activation.std().item(),
            "min": activation.min().item(),
            "max": activation.max().item(),
            "num_zeros": (activation == 0).sum().item(),
            "num_non_zeros": (activation != 0).sum().item(),
            "fraction_zeros": ((activation == 0).sum() / activation.numel()).item(),
        }

    def get_activation(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No activation data available for layer '{layer_name}'."
        return self.wrapped_model.current_state[layer_name]['output']

# ./inspection/variable_inspector.py
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

# ./inspection/model_inspector.py
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

# ./inspection/gradient_inspector.py
import torch
from core.model_wrapper import ModelWrapper

class GradientInspector:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def inspect(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No gradient data available for layer '{layer_name}'."

        grad_info = {}
        if 'grad_input' in self.wrapped_model.current_state[layer_name]:
            grad_input = self.wrapped_model.current_state[layer_name]['grad_input']
            grad_info['input_gradient'] = self._analyze_gradient(grad_input)

        if 'grad_output' in self.wrapped_model.current_state[layer_name]:
            grad_output = self.wrapped_model.current_state[layer_name]['grad_output']
            grad_info['output_gradient'] = self._analyze_gradient(grad_output)

        if 'grad_params' in self.wrapped_model.current_state[layer_name]:
            grad_params = self.wrapped_model.current_state[layer_name]['grad_params']
            grad_info['parameter_gradients'] = {name: self._analyze_gradient(grad) for name, grad in grad_params.items()}

        return grad_info

    def _analyze_gradient(self, gradient):
        if not isinstance(gradient, torch.Tensor):
            return "Gradient is not a tensor."

        return {
            "shape": gradient.shape,
            "mean": gradient.mean().item(),
            "std": gradient.std().item(),
            "min": gradient.min().item(),
            "max": gradient.max().item(),
            "norm": gradient.norm().item(),
            "num_zeros": (gradient == 0).sum().item(),
            "num_non_zeros": (gradient != 0).sum().item(),
        }

    def get_gradient(self, layer_name: str, grad_type: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No gradient data available for layer '{layer_name}'."
        
        if grad_type == 'input':
            return self.wrapped_model.current_state[layer_name].get('grad_input')
        elif grad_type == 'output':
            return self.wrapped_model.current_state[layer_name].get('grad_output')
        elif grad_type == 'params':
            return self.wrapped_model.current_state[layer_name].get('grad_params')
        else:
            return f"Invalid gradient type '{grad_type}'. Choose from 'input', 'output', or 'params'."

# ./inspection/layer_inspector.py
import torch
from core.model_wrapper import ModelWrapper

class LayerInspector:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def inspect(self, layer_name: str):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        layer_info = {
            "name": layer_name,
            "type": type(layer).__name__,
            "parameters": self._get_parameters_info(layer),
            "input_shape": self._get_input_shape(layer_name),
            "output_shape": self._get_output_shape(layer_name)
        }
        return layer_info

    def _get_parameters_info(self, layer):
        return {name: {"shape": param.shape, "requires_grad": param.requires_grad}
                for name, param in layer.named_parameters()}

    def _get_input_shape(self, layer_name):
        if layer_name in self.wrapped_model.current_state:
            inputs = self.wrapped_model.current_state[layer_name]['input']
            return [tuple(input.shape) for input in inputs]
        return None

    def _get_output_shape(self, layer_name):
        if layer_name in self.wrapped_model.current_state:
            output = self.wrapped_model.current_state[layer_name]['output']
            return tuple(output.shape) if isinstance(output, torch.Tensor) else None
        return None

# ./inspection/attention_inspector.py
import torch
from core.model_wrapper import ModelWrapper

class AttentionInspector:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def inspect(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No attention data available for layer '{layer_name}'."

        layer_output = self.wrapped_model.current_state[layer_name]['output']
        
        # Check if the layer output is a tuple (common in transformer models)
        if isinstance(layer_output, tuple):
            # Assume the second element contains attention weights
            attention_weights = layer_output[1]
        elif isinstance(layer_output, torch.Tensor):
            # If it's a tensor, assume it's the attention weights directly
            attention_weights = layer_output
        else:
            return f"Unexpected output type for layer '{layer_name}': {type(layer_output)}"

        return self._analyze_attention(attention_weights)

    def _analyze_attention(self, attention_weights: torch.Tensor):
        if len(attention_weights.shape) != 4:
            return f"Unexpected shape for attention weights: {attention_weights.shape}"

        batch_size, num_heads, seq_len, _ = attention_weights.shape

        return {
            "shape": attention_weights.shape,
            "num_heads": num_heads,
            "sequence_length": seq_len,
            "mean": attention_weights.mean().item(),
            "std": attention_weights.std().item(),
            "min": attention_weights.min().item(),
            "max": attention_weights.max().item(),
            "entropy": self._compute_attention_entropy(attention_weights),
            "top_k_attention": self._get_top_k_attention(attention_weights, k=5)
        }

    def _compute_attention_entropy(self, attention_weights: torch.Tensor):
        # Compute entropy of attention distribution
        attention_probs = attention_weights.mean(dim=1)  # Average over heads
        entropy = -(attention_probs * torch.log(attention_probs + 1e-9)).sum(dim=-1).mean().item()
        return entropy

    def _get_top_k_attention(self, attention_weights: torch.Tensor, k: int):
        # Get top-k attended positions
        mean_attention = attention_weights.mean(dim=(0, 1))  # Average over batch and heads
        top_k_values, top_k_indices = torch.topk(mean_attention, k)
        return [(idx.item(), val.item()) for idx, val in zip(top_k_indices, top_k_values)]

    def get_attention_weights(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No attention data available for layer '{layer_name}'."
        
        layer_output = self.wrapped_model.current_state[layer_name]['output']
        
        if isinstance(layer_output, tuple):
            return layer_output[1]
        elif isinstance(layer_output, torch.Tensor):
            return layer_output
        else:
            return f"Unexpected output type for layer '{layer_name}': {type(layer_output)}"

# ./cli/__init__.py
from .repl import NNGDBREPL
from .command_handler import CommandHandler
from .python_repl import PythonREPL

__all__ = ['NNGDBREPL', 'CommandHandler', 'PythonREPL']

# ./cli/repl.py
import readline
import rlcompleter
from .command_handler import CommandHandler

class NNGDBREPL:
    def __init__(self, debugger):
        self.debugger = debugger
        self.command_handler = CommandHandler(debugger)
        self.command_history = []
        self.setup_readline()

    def setup_readline(self):
        readline.parse_and_bind("tab: complete")
        readline.set_completer(rlcompleter.Completer(self.__dict__).complete)

    def run(self):
        print("Welcome to NNGDB (Neural Network GDB) v.0.0.1a")
        print("https://github.com/juvi21/nngdb")
        print("Type 'help' for a list of commands, or 'quit' to exit.")
        print("Preparing neural spells..")
        
        while True:
            current_experiment = self.debugger.get_current_experiment()
            try:
                user_input = input(f"<nngdb:{current_experiment}> ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                self.command_history.append(user_input)
                command_parts = user_input.split()
                if command_parts:
                    command, args = command_parts[0], command_parts[1:]
                    result = self.command_handler.handle_command(command, args)
                    print(result)
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
        
        print("Exiting NNGDB...")

    def get_command_completions(self, text, state):
        commands = self.command_handler.get_available_commands()
        matches = [cmd for cmd in commands if cmd.startswith(text)]
        return matches[state] if state < len(matches) else None
    
    def get_command_history(self):
        return self.command_history
    
    def complete(self, text, state):
        options = [cmd for cmd in self.command_handler.get_available_commands() if cmd.startswith(text)]
        if state < len(options):
            return options[state]
        else:
            return None
        
    def cmd_history(self, *args):
        """
        Show command history.
        Usage: history [n]
        """
        n = int(args[0]) if args else len(self.command_history)
        return "\n".join(self.command_history[-n:])
    
    def cmd_connect(self, *args):
        """
        Connect to a running NNGDB server.
        Usage: connect [host] [port]
        """
        host = args[0] if len(args) > 0 else 'localhost'
        port = int(args[1]) if len(args) > 1 else 5000
        self.debugger.connect(host, port)
        return f"Connected to NNGDB server at {host}:{port}"

# ./cli/python_repl.py
import code
import readline
import rlcompleter

class PythonREPL:
    def __init__(self, debugger):
        self.debugger = debugger

    def run(self):
        print("Entering Python REPL. Use 'debugger' to access the NNGDB instance.")
        print("Type 'exit()' or press Ctrl+D to return to NNGDB.")

        # Set up readline with tab completion
        readline.parse_and_bind("tab: complete")
        
        # Create a dictionary of local variables for the interactive console
        local_vars = {'debugger': self.debugger}
        
        # Start the interactive console
        code.InteractiveConsole(local_vars).interact(banner="")

# ./cli/command_handler.py
from typing import List
from core.debugger import NNGDB
from utils.error_handling import handle_exceptions
from transformers import AutoTokenizer
import torch
import cloudpickle

class CommandHandler:
    def __init__(self, debugger: NNGDB):
        self.debugger = debugger
        self.command_shortforms = {
            'c': 'continue',
            'h': 'help',
            'r': 'run',
            's': 'step',
            't': 'trace',
            'b': 'breakpoint',
            'a': 'analyze',
            'i': 'inspect',
            'm': 'modify',
            'l': 'log',
            'p': 'python',
            'e': 'experiment',
            'u': 'undo',
            'd': 'redo',
            'resetw': 'reset_weights',
            'ho': 'hook',
            'pr': 'probe'
        }

    @handle_exceptions
    def cmd_run(self, *args):
        """
        Run the model with the given input.
        Usage: run <input_text>
        """
        if not args:
            return "Error: No input provided. Usage: run <input_text>"
        input_text = " ".join(args)
        return self.debugger.run(input_text)

    @handle_exceptions
    def handle_command(self, command: str, args: List[str]) -> str:
        command = self.command_shortforms.get(command, command)
        
        method_name = f"cmd_{command}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(*args)
        else:
            return f"Unknown command: {command}. Type 'help' to see the list of available commands."

    @handle_exceptions
    def get_available_commands(self) -> List[str]:
        return [method[4:] for method in dir(self) if method.startswith("cmd_")]

    @handle_exceptions
    def cmd_probe(self, *args):
        """
        Manage probes and retrieve activations in the model.
        Usage: 
            probe add <point_name> <probe_function>
            probe clear
            probe list
            probe get <layer_name> <input_text>
            probe get_multiple <layer1>,<layer2>,... <input_text>
        Examples:
            probe add layers.4.mlp.post_activation "lambda save_ctx, tensor: save_ctx.tensor = tensor"
            probe clear
            probe list
            probe get layers.4.mlp.post_activation "Hello, world!"
            probe get_multiple layers.0.mlp.post_activation,layers.1.mlp.post_activation "Hello, world!"
        """
        if not args:
            return "Error: Insufficient arguments. Use 'help probe' for usage information."

        action = args[0]
        if action == "add":
            return self._probe_add(args[1:])
        elif action == "clear":
            return self._probe_clear()
        elif action == "list":
            return self._probe_list()
        elif action == "get":
            return self._probe_get(args[1:])
        elif action == "get_multiple":
            return self._probe_get_multiple(args[1:])
        else:
            return f"Unknown probe action: {action}. Valid actions are: add, clear, list, get, get_multiple"

    def _probe_add(self, args):
        if len(args) < 2:
            return "Error: Insufficient arguments. Usage: probe add <point_name> <probe_function>"
        
        point_name = args[0]
        probe_function = " ".join(args[1:])
        
        try:
            probe_func = cloudpickle.loads(cloudpickle.dumps(eval(f"lambda save_ctx, tensor: {probe_function}")))
        except Exception as e:
            return f"Error creating probe function: {str(e)}"

        return self.debugger.add_probe(point_name, probe_func)

    def _probe_clear(self):
        return self.debugger.clear_probes()

    def _probe_list(self):
        probes = self.debugger.list_probes()
        if not probes:
            return "No active probes."
        return "\n".join(f"{point}: {func}" for point, func in probes.items())

    def _probe_get(self, args):
        if len(args) < 2:
            return "Error: Insufficient arguments. Usage: probe get <layer_name> <input_text>"
        layer_name, input_text = args[0], " ".join(args[1:])
        activation = self.debugger.get_activation(layer_name, input_text)
        return f"Activation for {layer_name}:\n{activation}"

    def _probe_get_multiple(self, args):
        if len(args) < 2:
            return "Error: Insufficient arguments. Usage: probe get_multiple <layer1>,<layer2>,... <input_text>"
        layer_names, input_text = args[0].split(','), " ".join(args[1:])
        activations = self.debugger.get_multiple_activations(layer_names, input_text)
        result = "Activations:\n"
        for layer, activation in activations.items():
            result += f"{layer}:\n{activation}\n\n"
        return result
    
    @handle_exceptions
    def cmd_help(self, *args):
        command_descriptions = {
            'run': "Run the model with the given input. Usage: run <input_text>",
            'continue': "Continue execution after hitting a breakpoint. Usage: continue",
            'step': "Step through execution. Usage: step [<num_steps>]",
            'trace': "Manage tracing of execution, activations, or gradients. Usage: trace start|get|clear <type> [<layer_name>]",
            'breakpoint': "Set or remove a breakpoint. Usage: breakpoint set|remove|list <layer_name> [<condition>]",
            'analyze': "Perform various analyses on the model. Usage: analyze tokens|attention_representation|gradients|attention|activations <args>",
            'inspect': "Inspect a layer, weight, activation, or gradient. Usage: inspect <type> <name>",
            'modify': "Modify weights or activations in the model. Usage: modify weight|activation <layer_name> <details>",
            'log': "Log information or export logs. Usage: log info|warning|error|export <message>",
            'python': "Enter Python REPL for custom analysis. Usage: python",
            'experiment': "Manage and compare experiments. Usage: experiment create|switch|list|delete|compare|current <args>",
            'undo': "Undo the last action. Usage: undo",
            'redo': "Redo the last undone action. Usage: redo",
            'reset_weights': "Reset all modified weights to their original values. Usage: reset_weights",
            'hook': "Manage hooks in the model. Usage: hook add|remove|list|clear <args>",
            'help': "Show help for commands. Usage: help [<command>]"
        }
        
        if not args:
            commands = self.get_available_commands()
            help_text = "Available commands (short forms in parentheses):\n"
            for cmd in commands:
                short_form = next((key for key, value in self.command_shortforms.items() if value == cmd), None)
                help_text += f"{cmd} ({short_form}): {command_descriptions.get(cmd, 'No description available.')}\n"
            return help_text
        else:
            method_name = f"cmd_{args[0]}"
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                return method.__doc__ or "No help available for this command."
            else:
                return f"Unknown command: {args[0]}. Type 'help' to see the list of available commands."

    @handle_exceptions
    def cmd_inspect(self, *args):
        """
        Inspect a layer, weight, activation, or gradient.
        Usage: inspect <type> <name>
        Types: model, layer, weight, activation, gradient
        """
        if len(args) < 1:
            return "Error: Missing type argument. Usage: inspect <type> [<name>]"
        inspect_type = args[0]
        if inspect_type == "model":
            return self.debugger.inspect_model()
        elif len(args) < 2:
            return "Error: Missing name argument. Usage: inspect <type> <name>"
        name = args[1]
        if inspect_type == "layer":
            return self.debugger.inspect_layer(name)
        elif inspect_type == "weight":
            return self.debugger.inspect_weights(name)
        elif inspect_type == "activation":
            return self.debugger.inspect_activations(name)
        elif inspect_type == "gradient":
            return self.debugger.inspect_gradients(name)
        else:
            return f"Unknown inspection type: {inspect_type}. Valid types are: model, layer, weight, activation, gradient."
    
    @handle_exceptions
    def cmd_breakpoint(self, *args):
        """
        Set or remove a breakpoint.
        Usage: breakpoint set <layer_name> [<condition>]
               breakpoint remove <layer_name>
               breakpoint list
        """
        if not args:
            return "Error: No action specified. Usage: breakpoint <set|remove|list> [<layer_name>] [<condition>]"
        action = args[0]
        if action == "set":
            if len(args) < 2:
                return "Error: Missing layer name. Usage: breakpoint set <layer_name> [<condition>]"
            layer_name = args[1]
            condition = " ".join(args[2:]) if len(args) > 2 else None
            return self.debugger.set_breakpoint(layer_name, condition)
        elif action == "remove":
            if len(args) < 2:
                return "Error: Missing layer name. Usage: breakpoint remove <layer_name>"
            layer_name = args[1]
            return self.debugger.remove_breakpoint(layer_name)
        elif action == "list":
            return self.debugger.list_breakpoints()
        else:
            return f"Unknown breakpoint action: {action}. Valid actions are: set, remove, list."
    
    @handle_exceptions
    def cmd_step(self, *args):
        """
        Step through execution.
        Usage: step [<num_steps>]
        """
        try:
            num_steps = int(args[0]) if args else 1
        except ValueError:
            return "Error: Invalid number of steps. Usage: step [<num_steps>]"
        return self.debugger.step(num_steps)

    @handle_exceptions
    def cmd_continue(self, *args):
        """
        Continue execution after hitting a breakpoint.
        Usage: continue
        """
        return self.debugger.continue_execution()

    @handle_exceptions
    def cmd_modify(self, *args):
        """
        Modify weights or activations in the model.
        Usage: 
            modify weight <layer_name> <weight_name> <indices> <value>
            modify activation <layer_name> <function>
        Examples:
            modify weight model.layers.0.self_attn.q_proj weight (0,0) 1.0
            modify activation model.layers.0.self_attn "x * 2"
        """
        if len(args) < 3:
            return "Error: Insufficient arguments. Usage: modify <type> <layer_name> <details>"
        
        modify_type = args[0]
        layer_name = args[1]
        
        if modify_type == "weight":
            if len(args) != 5:
                return "Error: Incorrect number of arguments for weight modification. Usage: modify weight <layer_name> <weight_name> <indices> <value>"
            weight_name = args[2]
            indices = eval(args[3])  # Be careful with eval, ensure proper input validation
            value = float(args[4])
            return self.debugger.modify_weight(layer_name, weight_name, indices, value)
        
        elif modify_type == "activation":
            function_str = " ".join(args[2:])
            return self.debugger.modify_activation(layer_name, function_str)
        
        else:
            return f"Unknown modification type: {modify_type}. Valid types are: weight, activation."
    
    @handle_exceptions
    def cmd_analyze(self, *args):
        """
        Perform various analyses on the model.
         Usage:
            analyze tokens <input_text> <analysis_type> [options]
            analyze attention_representation <input_text> [options]
            analyze gradients [<layer_name>]
            analyze attention [<layer_name>]
            analyze activations [<layer_name>]
            analyze dataset_examples (<input_file> | --inline) <layer1> [<layer2> ...] [--top <n>] [--neuron <idx>]
        """
        if not args:
            return "Error: No analysis type provided. Usage: analyze <subcommand> <args>"
    
        subcommand = args[0]
        if subcommand == "tokens":
            return self._analyze_tokens(args[1:])
        elif subcommand == "attention_representation":
            return self._analyze_attention_representation(args[1:])
        elif subcommand in ["gradients", "attention", "activations"]:
            return self._analyze_layer(subcommand, args[1:])
        elif subcommand == "dataset_examples":
            return self._analyze_dataset_examples(args[1:])
        else:
            return f"Unknown analyze subcommand: {subcommand}. Valid subcommands are: tokens, attention_representation, gradients, attention, activations, dataset_examples."    
    
    @handle_exceptions
    def _analyze_tokens(self, args):
        if len(args) < 2:
            return "Error: Insufficient arguments for token analysis. Usage: analyze tokens <input_text> <analysis_type> [options]"
        input_text = args[0]
        analysis_type = args[1]
        options = self._parse_options(args[2:])
        return self.debugger.analyze_tokens(input_text, analysis_type, **options)

    def _analyze_attention_representation(self, args):
        if len(args) < 1:
            return "Error: No input text provided. Usage: analyze attention_representation <input_text> [options]"
        input_text = args[0]
        options = self._parse_options(args[1:])
        return self.debugger.analyze_token_attention_and_representation(input_text, **options)

    def _analyze_layer(self, analysis_type, args):
        layer_name = args[0] if args else None
        if analysis_type == "gradients":
            return self.debugger.gradient_inspector.inspect(layer_name)
        elif analysis_type == "attention":
            return self.debugger.attention_inspector.inspect(layer_name)
        elif analysis_type == "activations":
            return self.debugger.activation_inspector.inspect(layer_name)

    @handle_exceptions
    def cmd_log(self, *args):
        """
        Log information or export logs.
        Usage: log <action> [<args>]
        Actions: info, warning, error, export
        """
        if not args:
            return "Error: No action specified. Usage: log <action> [<args>]"
        action = args[0]
        message = " ".join(args[1:])
        if action == "info":
            self.debugger.log_info(message)
            return "Info logged."
        elif action == "warning":
            self.debugger.log_warning(message)
            return "Warning logged."
        elif action == "error":
            self.debugger.log_error(message)
            return "Error logged."
        elif action == "export":
            return self.debugger.export_logs(message)
        else:
            return f"Unknown log action: {action}. Valid actions are: info, warning, error, export."

    @handle_exceptions
    def cmd_python(self, *args):
        """
        Enter Python REPL for custom analysis.
        Usage: python
        """
        return self.debugger.enter_python_repl()
    
    @handle_exceptions
    def cmd_trace(self, *args):
        """
        Manage tracing of execution, activations, or gradients.
        Usage:
            trace start <type>
            trace get <type> [<layer_name>]
            trace clear
        Types: execution, activations, gradients
        """
        if not args:
            return "Error: No trace action specified. Usage: trace <subcommand> <args>"

        subcommand = args[0]
        if subcommand == "start":
            return self._trace_start(args[1:])
        elif subcommand == "get":
            return self._trace_get(args[1:])
        elif subcommand == "clear":
            return self._trace_clear()
        else:
            return f"Unknown trace subcommand: {subcommand}. Valid subcommands are: start, get, clear."

    def _trace_start(self, args):
        if not args:
            return "Error: No trace type specified. Usage: trace start <type> (execution, activations, or gradients)"
        trace_type = args[0]
        if trace_type == "execution":
            return self.debugger.trace_execution()
        elif trace_type == "activations":
            return self.debugger.trace_activations()
        elif trace_type == "gradients":
            return self.debugger.trace_gradients()
        else:
            return f"Unknown trace type: {trace_type}. Valid types are: execution, activations, gradients."

    def _trace_get(self, args):
        if len(args) < 1:
            return "Error: No trace type specified. Usage: trace get <type> [<layer_name>]"
        trace_type = args[0]
        layer_name = args[1] if len(args) > 1 else None
        if trace_type == "execution":
            return self.debugger.get_execution_trace()
        elif trace_type == "activations":
            if layer_name is None:
                return "Error: No layer name specified. Usage: trace get activations <layer_name>"
            return self.debugger.get_activation_trace(layer_name)
        elif trace_type == "gradients":
            if layer_name is None:
                return "Error: No layer name specified. Usage: trace get gradients <layer_name>"
            return self.debugger.get_gradient_trace(layer_name)
        else:
            return f"Unknown trace type: {trace_type}. Valid types are: execution, activations, gradients."

    def _trace_clear(self):
        return self.debugger.clear_all_traces()
    
    @handle_exceptions
    def cmd_compare_tokens(self, *args):
        """
        Compare token probabilities between two analyses.
        Usage: compare_tokens <index1> <index2>
        """
        if len(args) != 2:
            return "Error: Incorrect number of indices provided. Usage: compare_tokens <index1> <index2>"
        try:
            index1, index2 = map(int, args)
            return self.debugger.compare_token_probabilities(index1, index2)
        except ValueError:
            return "Error: Invalid indices. Please provide two integer values."
    
    @handle_exceptions
    def cmd_undo(self, *args):
        """
        Undo the last action.
        Usage: undo
        """
        return self.debugger.undo()

    @handle_exceptions
    def cmd_redo(self, *args):
        """
        Redo the last undone action.
        Usage: redo
        """
        return self.debugger.redo()
    
    @handle_exceptions
    def cmd_reset_weights(self, *args):
        """
        Reset all modified weights to their original values.
        Usage: reset_weights
        """
        return self.debugger.reset_modified_weights()
    
    @handle_exceptions
    def cmd_hook(self, *args):
        """
        Manage hooks in the model.
        Usage:
            hook add <forward|backward> <module_name> <hook_name> <hook_function>
            hook remove <hook_name>
            hook list
            hook clear
        """
        if not args:
            return "Error: No subcommand specified. Usage: hook <add|remove|list|clear> <args>"

        subcommand = args[0]
        if subcommand == "add":
            return self._hook_add(args[1:])
        elif subcommand == "remove":
            return self._hook_remove(args[1:])
        elif subcommand == "list":
            return self._hook_list()
        elif subcommand == "clear":
            return self._hook_clear()
        else:
            return f"Unknown hook subcommand: {subcommand}. Valid subcommands are: add, remove, list, clear."

    def _hook_add(self, args):
        if len(args) < 4:
            return "Error: Insufficient arguments for adding hook. Usage: hook add <forward|backward> <module_name> <hook_name> <hook_function>"
        hook_type, module_name, hook_name = args[0], args[1], args[2]
        hook_function = " ".join(args[3:])
        return self.debugger.add_hook(hook_type, module_name, hook_name, hook_function)

    def _hook_remove(self, args):
        if not args:
            return "Error: No hook name provided. Usage: hook remove <hook_name>"
        hook_name = args[0]
        return self.debugger.remove_hook(hook_name)

    def _hook_list(self):
        return self.debugger.list_hooks()

    def _hook_clear(self):
        return self.debugger.clear_hooks()
    
    @handle_exceptions
    def _parse_options(self, args):
        options = {}
        for i in range(0, len(args), 2):
            if args[i] == '--compare_modified':
                options['compare_modified'] = True
                continue
            if i + 1 < len(args):
                key = args[i].lstrip('-')
                value = args[i + 1]
                if value.isdigit():
                    options[key] = int(value)
                elif value.lower() in ['true', 'false']:
                    options[key] = value.lower() == 'true'
                elif ',' in value:
                    options[key] = value.split(',')
                else:
                    options[key] = value
        return options
    
    @handle_exceptions
    def cmd_experiment(self, *args):
        """
        Manage and compare experiments.
        Usage:
            experiment create <name>
            experiment switch <name>
            experiment list
            experiment delete <name>
            experiment compare <exp1> <exp2> <input_text> [analysis_type] [options]
            experiment current
        """
        if not args:
            return "Error: No subcommand specified. Usage: experiment <subcommand> <args>"

        subcommand = args[0]
        if subcommand == "create":
            return self._experiment_create(args[1:])
        elif subcommand == "switch":
            return self._experiment_switch(args[1:])
        elif subcommand == "list":
            return self._experiment_list(args[1:])
        elif subcommand == "delete":
            return self._experiment_delete(args[1:])
        elif subcommand == "compare":
            return self._experiment_compare(args[1:])
        elif subcommand == "current":
            return self._experiment_current(args[1:])
        else:
            return f"Unknown experiment subcommand: {subcommand}. Valid subcommands are: create, switch, list, delete, compare, current."
    
    @handle_exceptions
    def _experiment_create(self, args):
        """
        Create a new experiment.
        Usage: experiment create <name>
        """
        if len(args) != 1:
            return "Error: Missing experiment name. Usage: experiment create <name>"
        return self.debugger.create_experiment(args[0])

    @handle_exceptions
    def _experiment_switch(self, args):
        if len(args) != 1:
            return "Error: Missing experiment name. Usage: experiment switch <name>"
        result = self.debugger.switch_experiment(args[0])
        if result.startswith("Experiment"):  # Error message
            return result
        return {result}

    @handle_exceptions
    def _experiment_list(self, args):
        """
        List all available experiments.
        Usage: experiment list
        """
        experiments = self.debugger.list_experiments()
        if not experiments:
            return "No experiments available."
        return "Available experiments:\n" + "\n".join(experiments)

    @handle_exceptions
    def _experiment_delete(self, args):
        """
        Delete an existing experiment.
        Usage: experiment delete <name>
        """
        if len(args) != 1:
            return "Error: Missing experiment name. Usage: experiment delete <name>"
        return self.debugger.delete_experiment(args[0])

    @handle_exceptions
    def _experiment_compare(self, args):
        """
        Compare two experiments.
        Usage: experiment compare <exp1> <exp2> <input_text> [analysis_type] [options]
        """
        if len(args) < 3:
            return "Error: Insufficient arguments for comparison. Usage: experiment compare <exp1> <exp2> <input_text> [analysis_type] [options]"
        exp1, exp2 = args[0], args[1]
        input_text = args[2]
        analysis_type = args[3] if len(args) > 3 else 'probabilities'
        options = self._parse_options(args[4:])
        return self.debugger.compare_experiments(exp1, exp2, input_text, analysis_type, **options)

    @handle_exceptions
    def _experiment_current(self, args):
        """
        Show the current active experiment.
        Usage: experiment current
        """
        current_exp = self.debugger.get_current_experiment()
        return f"Current experiment: {current_exp}" if current_exp else "No active experiment."

    @handle_exceptions
    def _analyze_dataset_examples(self, args):
        if len(args) < 2:
            return "Error: Insufficient arguments. Usage: analyze dataset_examples (--inline | <input_file>) <layer1> [<layer2> ...] [--top <n>] [--neuron <idx>]"

        inline_mode = False
        input_source = None
        layer_names = []
        top_n = 10
        neuron_idx = None

        i = 0
        while i < len(args):
            if args[i] == '--inline':
                inline_mode = True
                i += 1
            elif args[i] == '--top':
                top_n = int(args[i+1])
                i += 2
            elif args[i] == '--neuron':
                neuron_idx = int(args[i+1])
                i += 2
            elif input_source is None and not inline_mode:
                input_source = args[i]
                i += 1
            else:
                layer_names.append(args[i])
                i += 1

        if inline_mode:
            input_texts = self._get_inline_dataset()
        elif input_source:
            try:
                with open(input_source, 'r') as f:
                    input_texts = f.readlines()
            except FileNotFoundError:
                return f"Error: Input file '{input_source}' not found."
        else:
            return "Error: No input source specified. Use --inline or provide an input file."

        result = self.debugger.collect_dataset_examples(input_texts, layer_names, top_n)
        
        if neuron_idx is not None:
            return self._format_neuron_examples(result, layer_names, neuron_idx)
        else:
            return self._format_layer_examples(result, layer_names)
    
    @handle_exceptions
    def _get_inline_dataset(self):
        print("Enter your dataset examples. Type 'END' on a new line when finished:")
        examples = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            examples.append(line)
        return examples

    @handle_exceptions
    def _format_neuron_examples(self, result, layer_names, neuron_idx):
        formatted = ""
        for layer in layer_names:
            if layer in result and neuron_idx < len(result[layer]):
                formatted += f"Top examples for layer '{layer}', neuron {neuron_idx}:\n"
                for activation, token in result[layer][neuron_idx]:
                    formatted += f"  {token}: {activation:.4f}\n"
            else:
                formatted += f"No examples found for layer '{layer}', neuron {neuron_idx}\n"
        return formatted

    @handle_exceptions
    def _format_layer_examples(self, result, layer_names):
        formatted = ""
        for layer in layer_names:
            if layer in result:
                formatted += f"Top activated neurons for layer '{layer}':\n"
                for neuron_idx, (activation, token) in enumerate(result[layer]):
                    formatted += f"  Neuron {neuron_idx}: {token} ({activation:.4f})\n"
            else:
                formatted += f"No examples found for layer '{layer}'\n"
        return formatted

# ./modification/__init__.py
from .weight_modifier import WeightModifier
from .activation_modifier import ActivationModifier
from .hyperparameter_modifier import HyperparameterModifier
from .model_surgery import ModelSurgeon

__all__ = ['WeightModifier', 'ActivationModifier', 'HyperparameterModifier', 'ModelSurgeon']

# ./modification/weight_modifier.py
import torch
from core.model_wrapper import ModelWrapper

class WeightModifier:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def modify_weight(self, layer_name: str, weight_name: str, indices, value):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, weight_name):
            return f"Weight '{weight_name}' not found in layer '{layer_name}'."

        weight = getattr(layer, weight_name)
        if not isinstance(weight, torch.Tensor):
            return f"'{weight_name}' is not a tensor in layer '{layer_name}'."

        try:
            with torch.no_grad():
                weight[indices] = value
            return f"Weight at {layer_name}.{weight_name}{indices} modified to {value}"
        except Exception as e:
            return f"Error modifying weight: {str(e)}"

    def scale_weights(self, layer_name: str, weight_name: str, scale_factor: float):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, weight_name):
            return f"Weight '{weight_name}' not found in layer '{layer_name}'."

        weight = getattr(layer, weight_name)
        if not isinstance(weight, torch.Tensor):
            return f"'{weight_name}' is not a tensor in layer '{layer_name}'."

        with torch.no_grad():
            weight.mul_(scale_factor)

        return f"Weights in {layer_name}.{weight_name} scaled by {scale_factor}"

    def reset_weights(self, layer_name: str):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        def weight_reset(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        layer.apply(weight_reset)
        return f"Weights in layer '{layer_name}' have been reset."

    def add_noise_to_weights(self, layer_name: str, weight_name: str, noise_scale: float):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, weight_name):
            return f"Weight '{weight_name}' not found in layer '{layer_name}'."

        weight = getattr(layer, weight_name)
        if not isinstance(weight, torch.Tensor):
            return f"'{weight_name}' is not a tensor in layer '{layer_name}'."

        with torch.no_grad():
            noise = torch.randn_like(weight) * noise_scale
            weight.add_(noise)

        return f"Noise added to weights in {layer_name}.{weight_name} with scale {noise_scale}"

    def prune_weights(self, layer_name: str, weight_name: str, threshold: float):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, weight_name):
            return f"Weight '{weight_name}' not found in layer '{layer_name}'."

        weight = getattr(layer, weight_name)
        if not isinstance(weight, torch.Tensor):
            return f"'{weight_name}' is not a tensor in layer '{layer_name}'."

        with torch.no_grad():
            mask = (weight.abs() > threshold).float()
            weight.mul_(mask)

        pruned_percentage = (1 - mask.mean().item()) * 100
        return f"Pruned {pruned_percentage:.2f}% of weights in {layer_name}.{weight_name}"

# ./modification/hyperparameter_modifier.py
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

# ./modification/model_surgery.py
import torch
import torch.nn as nn
from core.model_wrapper import ModelWrapper

class ModelSurgeon:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def replace_layer(self, layer_name: str, new_layer: nn.Module):
        parent_name, child_name = layer_name.rsplit('.', 1)
        parent_module = self.wrapped_model.get_layer(parent_name)

        if parent_module is None:
            return f"Parent module of '{layer_name}' not found."

        if not hasattr(parent_module, child_name):
            return f"Layer '{child_name}' not found in '{parent_name}'."

        setattr(parent_module, child_name, new_layer)
        return f"Layer '{layer_name}' replaced with {type(new_layer).__name__}"

    def insert_layer(self, layer_name: str, new_layer: nn.Module, position: str = 'after'):
        parent_name, child_name = layer_name.rsplit('.', 1)
        parent_module = self.wrapped_model.get_layer(parent_name)

        if parent_module is None:
            return f"Parent module of '{layer_name}' not found."

        if not hasattr(parent_module, child_name):
            return f"Layer '{child_name}' not found in '{parent_name}'."

        original_layer = getattr(parent_module, child_name)
        
        class WrappedLayer(nn.Module):
            def __init__(self, original_layer, new_layer, position):
                super().__init__()
                self.original_layer = original_layer
                self.new_layer = new_layer
                self.position = position

            def forward(self, x):
                if self.position == 'before':
                    x = self.new_layer(x)
                    return self.original_layer(x)
                elif self.position == 'after':
                    x = self.original_layer(x)
                    return self.new_layer(x)

        wrapped_layer = WrappedLayer(original_layer, new_layer, position)
        setattr(parent_module, child_name, wrapped_layer)

        return f"Layer '{new_layer.__class__.__name__}' inserted {position} '{layer_name}'"

    def remove_layer(self, layer_name: str):
        parent_name, child_name = layer_name.rsplit('.', 1)
        parent_module = self.wrapped_model.get_layer(parent_name)

        if parent_module is None:
            return f"Parent module of '{layer_name}' not found."

        if not hasattr(parent_module, child_name):
            return f"Layer '{child_name}' not found in '{parent_name}'."

        class Identity(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        setattr(parent_module, child_name, Identity())
        return f"Layer '{layer_name}' removed and replaced with Identity"

    def change_activation_function(self, layer_name: str, new_activation: nn.Module):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        if not hasattr(layer, 'activation'):
            return f"Layer '{layer_name}' does not have an 'activation' attribute."

        layer.activation = new_activation
        return f"Activation function of '{layer_name}' changed to {type(new_activation).__name__}"

# ./modification/activation_modifier.py
import torch
from core.model_wrapper import ModelWrapper

class ActivationModifier:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.hooks = {}

    def modify_activation(self, layer_name: str, modification_function):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."

        def hook(module, input, output):
            return modification_function(output)

        handle = layer.register_forward_hook(hook)
        self.hooks[layer_name] = handle

        return f"Activation modification hook set for layer '{layer_name}'"

    def remove_modification(self, layer_name: str):
        if layer_name in self.hooks:
            self.hooks[layer_name].remove()
            del self.hooks[layer_name]
            return f"Activation modification removed for layer '{layer_name}'"
        else:
            return f"No activation modification found for layer '{layer_name}'"

    def clear_all_modifications(self):
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
        return "All activation modifications cleared"

    def add_noise_to_activation(self, layer_name: str, noise_scale: float):
        def add_noise(output):
            return output + torch.randn_like(output) * noise_scale

        return self.modify_activation(layer_name, add_noise)

    def clip_activation(self, layer_name: str, min_val: float, max_val: float):
        def clip(output):
            return torch.clamp(output, min_val, max_val)

        return self.modify_activation(layer_name, clip)

    def scale_activation(self, layer_name: str, scale_factor: float):
        def scale(output):
            return output * scale_factor

        return self.modify_activation(layer_name, scale)

