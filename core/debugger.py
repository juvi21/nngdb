import torch
from typing import Any, Dict, Optional
from transformers import AutoTokenizer
import struct

from utils.error_handling import handle_exceptions
from .model_wrapper import ModelWrapper
from .execution_engine import ExecutionEngine
from inspection import ModelInspector, LayerInspector, WeightInspector, ActivationInspector, GradientInspector, AttentionInspector, VariableInspector
from breakpoints import BreakpointManager
from tracing import ExecutionTracer, ActivationTracer, GradientTracer
from analysis.token_probability import TokenProbabilityAnalyzer
from analysis.token_analyzer import TokenAnalyzer
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

    def _initialize_dummy_components(self):
        # Initialize dummy components or set to None for remote execution
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
        self.send_msg(pickle.dumps(command))
        return pickle.loads(self.recv_msg())

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

            return f"Input: {input_text}\nOutput: {output_text}"
        except Exception as e:
            return f"Error: {str(e)}\nError type: {type(e)}"

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
    def trace_execution(self):
        if self.socket:
            return self.execute_remote('trace_execution')
        return self.execution_tracer.trace()

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
    def step(self, num_steps: int = 1):
        if self.socket:
            return self.execute_remote('step', num_steps)
        return self.execution_engine.step(num_steps)
    
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