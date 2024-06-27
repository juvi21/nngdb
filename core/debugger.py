import torch
from typing import Any, Dict
from transformers import AutoTokenizer

from .model_wrapper import ModelWrapper
from .execution_engine import ExecutionEngine
from inspection import ModelInspector, LayerInspector, WeightInspector, ActivationInspector, GradientInspector, AttentionInspector, VariableInspector
from breakpoints import BreakpointManager
from tracing import ExecutionTracer, ActivationTracer, GradientTracer
from analysis.token_probability import TokenProbabilityAnalyzer
from core.undo_manager import UndoManager

class NNGDB:
    def __init__(self, model: torch.nn.Module, model_name: str, device: str):
        self.wrapped_model = ModelWrapper(model)
        self.execution_engine = ExecutionEngine(self.wrapped_model)
        self.breakpoint_manager = BreakpointManager(self.wrapped_model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        # Inspectors
        self.model_inspector = ModelInspector(self.wrapped_model)
        self.layer_inspector = LayerInspector(self.wrapped_model)
        self.weight_inspector = WeightInspector(self.wrapped_model)
        self.activation_inspector = ActivationInspector(self.wrapped_model)
        self.gradient_inspector = GradientInspector(self.wrapped_model)
        self.attention_inspector = AttentionInspector(self.wrapped_model)
        self.variable_inspector = VariableInspector(self.wrapped_model)
        
        # Tracers
        self.execution_tracer = ExecutionTracer(self.wrapped_model)
        self.activation_tracer = ActivationTracer(self.wrapped_model)
        self.gradient_tracer = GradientTracer(self.wrapped_model)
        
        self.context: Dict[str, Any] = {}

        self.undo_manager = UndoManager()
        self.token_analyzer = TokenProbabilityAnalyzer(model, self.tokenizer)

    
    def analyze_token_probabilities(self, input_text, top_k=5):
        if self.tokenizer is None:
            return "Error: Tokenizer not initialized"
        try:
            result = self.token_analyzer.analyze(input_text, top_k)
            return f"Input: {input_text}\nTop {top_k} tokens:\n" + "\n".join([f"{token}: {prob:.4f}" for token, prob in result['top_tokens']])
        except Exception as e:
            return f"Error analyzing tokens: {str(e)}"

    def compare_token_probabilities(self, index1, index2):
        return self.token_analyzer.compare(index1, index2)
    
    def analyze_tokens(self, input_text: str, top_k: int = 5):
        if self.tokenizer is None:
            return "Error: Tokenizer not initialized"
        return self.token_analyzer.analyze(input_text, top_k)

    def undo(self):
        state = self.undo_manager.undo()
        if state:
            self.wrapped_model.load_state_dict(state)
            return "Undo successful"
        return "Nothing to undo"

    def redo(self):
        state = self.undo_manager.redo()
        if state:
            self.wrapped_model.load_state_dict(state)
            return "Redo successful"
        return "Nothing to redo"

    def run(self, input_text: str):
        try:
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            print(f"Input shape: {inputs.input_ids.shape}")

            # Generate output
            with torch.no_grad():
                outputs = self.wrapped_model.model(**inputs)

            print(f"Output type: {type(outputs)}")
            if hasattr(outputs, 'logits'):
                print(f"Logits shape: {outputs.logits.shape}")
            
            # Process output
            if hasattr(outputs, 'logits'):
                output_ids = outputs.logits[:, -1:].argmax(dim=-1)
            elif isinstance(outputs, tuple):
                output_ids = outputs[0][:, -1:]
            else:
                output_ids = outputs[:, -1:]

            print(f"Output IDs shape: {output_ids.shape}")

            # Decode output
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            return f"Input: {input_text}\nOutput: {output_text}"
        except Exception as e:
            return f"Error: {str(e)}\nError type: {type(e)}"
    def set_context(self, key: str, value: Any):
        self.context[key] = value

    def get_context(self, key: str) -> Any:
        return self.context.get(key)

    def inspect_model(self):
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

    def inspect_layer(self, layer_name: str):
        return self.layer_inspector.inspect(layer_name)

    def inspect_weights(self, layer_name: str):
        return self.weight_inspector.inspect(layer_name)

    def inspect_activations(self, layer_name: str):
        return self.activation_inspector.inspect(layer_name)

    def inspect_gradients(self, layer_name: str):
        return self.gradient_inspector.inspect(layer_name)

    def inspect_attention(self, layer_name: str):
        return self.attention_inspector.inspect(layer_name)

    def inspect_variable(self, variable_name: str):
        return self.variable_inspector.inspect(variable_name)

    def set_breakpoint(self, layer_name: str, condition: str = None):
        return self.breakpoint_manager.set_breakpoint(layer_name, condition)

    def remove_breakpoint(self, layer_name: str):
        return self.breakpoint_manager.remove_breakpoint(layer_name)

    def list_breakpoints(self):
        return self.breakpoint_manager.list_breakpoints()

    def trace_execution(self):
        return self.execution_tracer.trace()

    def trace_activations(self):
        return self.activation_tracer.trace()

    def trace_gradients(self):
        return self.gradient_tracer.trace()
    
    def get_activation_trace(self, layer_name: str):
        return self.activation_tracer.get_activation(layer_name)
    
    def get_execution_trace(self):
        return self.execution_tracer.get_trace()
    
    def clear_all_traces(self):
        self.execution_tracer.clear_trace()
        self.activation_tracer.clear_trace()
        self.gradient_tracer.clear_trace()
        return "All traces cleared."
    
    def step(self, num_steps: int = 1):
        return self.execution_engine.step(num_steps)
    
    def continue_execution(self):
        return self.execution_engine.continue_execution()
    
    def get_token_attention(self, layer_name: str, head_index: int):
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None:
            return f"Layer '{layer_name}' not found."
        
        if not hasattr(layer, 'self_attn'):
            return f"Layer '{layer_name}' does not have attention weights."
        
        attention = layer.self_attn.attention_weights
        if attention is None:
            return f"No attention weights available for layer '{layer_name}'."
        
        if head_index >= attention.size(1):
            return f"Invalid head index. Layer '{layer_name}' has {attention.size(1)} attention heads."
        
        head_attention = attention[0, head_index].cpu().detach().numpy()
        return head_attention

    def get_token_representation(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No data available for layer '{layer_name}'."
        
        hidden_states = self.wrapped_model.current_state[layer_name]['output'][0]
        return hidden_states.cpu().detach().numpy()
    
    def get_gradient_trace(self, layer_name: str):
        return self.gradient_tracer.get_gradient(layer_name)
    
    def modify_weight(self, layer_name: str, weight_name: str, indices, value):
        return self.wrapped_model.modify_weight(layer_name, weight_name, indices, value)

    def modify_activation(self, layer_name: str, function_str: str):
        try:
            result = self.wrapped_model.modify_activation(layer_name, function_str)
            return result
        except Exception as e:
            return f"Error modifying activation: {str(e)}"

    def analyze_tokens_with_modified_weights(self, input_text, top_k=5):
        original_result = self.analyze_token_probabilities(input_text, top_k)
        
        # Ensure the input is on the correct device
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.wrapped_model.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        probs = torch.softmax(logits[0, -1], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        modified_result = {
            "input_text": input_text,
            "top_tokens": [
                (self.tokenizer.decode([idx.item()]), prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
        }
        
        comparison = f"Original analysis:\n{original_result}\n\n"
        comparison += f"Analysis with modified weights:\n"
        comparison += f"Input: {input_text}\nTop {top_k} tokens:\n" + "\n".join([f"{token}: {prob:.4f}" for token, prob in modified_result['top_tokens']])
        
        return comparison

    def reset_modified_weights(self):
        return self.wrapped_model.reset_modified_weights()
        
    def get_layer_representation(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No data available for layer '{layer_name}'."
        return self.wrapped_model.current_state[layer_name]['output']