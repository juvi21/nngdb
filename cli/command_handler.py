from typing import List
from core.debugger import NNGDB
from advanced.interpretability_metrics import InterpretabilityMetrics
from advanced.explainability_techniques import ExplainabilityTechniques
from utils.error_handling import handle_exceptions
from transformers import AutoTokenizer
import torch

class CommandHandler:
    def __init__(self, debugger: NNGDB):
        self.debugger = debugger

    @handle_exceptions
    def cmd_run(self, *args):
        """
        Run the model with the given input.
        Usage: run <input_text>
        """
        if not args:
            return "Usage: run <input_text>"
        input_text = " ".join(args)
        return self.debugger.run(input_text)

    @handle_exceptions
    def handle_command(self, command: str, args: List[str]) -> str:
        method_name = f"cmd_{command}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(*args)
        else:
            return f"Unknown command: {command}"
    
    @handle_exceptions
    def get_available_commands(self) -> List[str]:
        return [method[4:] for method in dir(self) if method.startswith("cmd_")]
    
    @handle_exceptions
    def cmd_help(self, *args):
        if not args:
            commands = self.get_available_commands()
            return "Available commands:\n" + "\n".join(commands)
        else:
            method_name = f"cmd_{args[0]}"
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                return method.__doc__ or "No help available for this command."
            else:
                return f"Unknown command: {args[0]}"

    @handle_exceptions
    def cmd_inspect(self, *args):
        """
        Inspect a layer, weight, activation, or gradient.
        Usage: inspect <type> <name>
        Types: model, layer, weight, activation, gradient
        """
        if len(args) < 1:
            return "Usage: inspect <type> [<name>]"
        inspect_type = args[0]
        if inspect_type == "model":
            return self.debugger.inspect_model()
        elif len(args) < 2:
            return "Usage: inspect <type> <name>"
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
            return f"Unknown inspection type: {inspect_type}"
    
    @handle_exceptions
    def cmd_breakpoint(self, *args):
        """
        Set or remove a breakpoint.
        Usage: breakpoint set <layer_name> [<condition>]
               breakpoint remove <layer_name>
               breakpoint list
        """
        if not args:
            return "Usage: breakpoint <set|remove|list> [<layer_name>] [<condition>]"
        action = args[0]
        if action == "set":
            if len(args) < 2:
                return "Usage: breakpoint set <layer_name> [<condition>]"
            layer_name = args[1]
            condition = " ".join(args[2:]) if len(args) > 2 else None
            return self.debugger.set_breakpoint(layer_name, condition)
        elif action == "remove":
            if len(args) < 2:
                return "Usage: breakpoint remove <layer_name>"
            layer_name = args[1]
            return self.debugger.remove_breakpoint(layer_name)
        elif action == "list":
            return self.debugger.list_breakpoints()
        else:
            return f"Unknown breakpoint action: {action}"
    
    @handle_exceptions
    def cmd_step(self, *args):
        """
        Step through execution.
        Usage: step [<num_steps>]
        """
        num_steps = int(args[0]) if args else 1
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
            return "Usage: modify <type> <layer_name> <details>"
        
        modify_type = args[0]
        layer_name = args[1]
        
        if modify_type == "weight":
            if len(args) != 5:
                return "Usage: modify weight <layer_name> <weight_name> <indices> <value>"
            weight_name = args[2]
            indices = eval(args[3])  # Be careful with eval, ensure proper input validation
            value = float(args[4])
            return self.debugger.modify_weight(layer_name, weight_name, indices, value)
        
        elif modify_type == "activation":
            function_str = " ".join(args[2:])
            return self.debugger.modify_activation(layer_name, function_str)
        
        else:
            return f"Unknown modification type: {modify_type}"
    
    @handle_exceptions
    def cmd_analyze(self, *args):
        """
        Perform analysis on the model.
        Usage: analyze <type> [<args>]
        Types: gradients, attention, activations
        """
        if not args:
            return "Usage: analyze <type> [<args>]"
        analysis_type = args[0]
        if analysis_type == "gradients":
            return self.debugger.analyze_gradients()
        elif analysis_type == "attention":
            return self.debugger.analyze_attention()
        elif analysis_type == "activations":
            return self.debugger.analyze_activations()
        else:
            return f"Unknown analysis type: {analysis_type}"
    
    @handle_exceptions
    def cmd_log(self, *args):
        """
        Log information or export logs.
        Usage: log <action> [<args>]
        Actions: info, warning, error, export
        """
        if not args:
            return "Usage: log <action> [<args>]"
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
            return f"Unknown log action: {action}"

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
        Start tracing execution, activations, or gradients.
        Usage: trace <type>
        Types: execution, activations, gradients
        """
        if not args:
            return "Usage: trace <type> (execution, activations, or gradients)"
        trace_type = args[0]
        if trace_type == "execution":
            return self.debugger.trace_execution()
        elif trace_type == "activations":
            return self.debugger.trace_activations()
        elif trace_type == "gradients":
            return self.debugger.trace_gradients()
        else:
            return f"Unknown trace type: {trace_type}"
    
    @handle_exceptions
    def cmd_get_trace(self, *args):
        """
        Get the trace for execution, activations, or gradients.
        Usage: get_trace <type> [<layer_name>]
        Types: execution, activations, gradients
        """
        if not args:
            return "Usage: get_trace <type> [<layer_name>]"
        trace_type = args[0]
        if trace_type == "execution":
            return self.debugger.get_execution_trace()
        elif trace_type == "activations":
            if len(args) < 2:
                return "Usage: get_trace activations <layer_name>"
            layer_name = args[1]
            return self.debugger.get_activation_trace(layer_name)
        elif trace_type == "gradients":
            if len(args) < 2:
                return "Usage: get_trace gradients <layer_name>"
            layer_name = args[1]
            return self.debugger.get_gradient_trace(layer_name)
        else:
            return f"Unknown trace type: {trace_type}"
    
    @handle_exceptions
    def cmd_clear_traces(self, *args):
        """
        Clear all traces.
        Usage: clear_traces
        """
        return self.debugger.clear_all_traces()
    
    @handle_exceptions
    def cmd_analyze_tokens(self, *args):
        """
        Analyze token probabilities for the given input.
        Usage: analyze_tokens <input_text> [top_k]
        """
        if not args:
            return "Usage: analyze_tokens <input_text> [top_k]"
        input_text = " ".join(args[:-1]) if len(args) > 1 and args[-1].isdigit() else " ".join(args)
        top_k = int(args[-1]) if len(args) > 1 and args[-1].isdigit() else 5
        return self.debugger.analyze_token_probabilities(input_text, top_k)
    
    @handle_exceptions
    def cmd_compare_tokens(self, *args):
        """
        Compare token probabilities between two analyses.
        Usage: compare_tokens <index1> <index2>
        """
        if len(args) != 2:
            return "Usage: compare_tokens <index1> <index2>"
        try:
            index1, index2 = map(int, args)
            return self.debugger.compare_token_probabilities(index1, index2)
        except ValueError:
            return "Invalid indices. Please provide two integer values."
    
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
    def cmd_token_attention(self, *args):
        """
        Get attention weights for a specific layer and attention head.
        Usage: token_attention <layer_name> <head_index>
        """
        if len(args) != 2:
            return "Usage: token_attention <layer_name> <head_index>"
        layer_name, head_index = args[0], int(args[1])
        return self.debugger.get_token_attention(layer_name, head_index)

    @handle_exceptions
    def cmd_token_representation(self, *args):
        """
        Get token representations for a specific layer.
        Usage: token_representation <layer_name>
        """
        if len(args) != 1:
            return "Usage: token_representation <layer_name>"
        layer_name = args[0]
        return self.debugger.get_token_representation(layer_name)
    
    @handle_exceptions
    def cmd_modify_weight(self, *args):
        """
        Modify a weight in the model.
        Usage: modify_weight <layer_name> <weight_name> <indices> <value>
        """
        if len(args) < 4:
            return "Usage: modify_weight <layer_name> <weight_name> <indices> <value>"
        layer_name = args[0]
        weight_name = args[1]
        indices = eval(args[2])  # Be careful with eval, ensure proper input validation
        value = float(args[3])
        return self.debugger.modify_weight(layer_name, weight_name, indices, value)
    
    @handle_exceptions
    def cmd_reset_weights(self, *args):
        """
        Reset all modified weights to their original values.
        Usage: reset_weights
        """
        return self.debugger.reset_modified_weights()

    @handle_exceptions
    def cmd_analyze_tokens_modified(self, *args):
        """
        Analyze token probabilities with modified weights and compare to original.
        Usage: analyze_tokens_modified <input_text> [top_k]
        """
        if not args:
            return "Usage: analyze_tokens_modified <input_text> [top_k]"
        input_text = " ".join(args[:-1]) if len(args) > 1 and args[-1].isdigit() else " ".join(args)
        top_k = int(args[-1]) if len(args) > 1 and args[-1].isdigit() else 5
        return self.debugger.analyze_tokens_with_modified_weights(input_text, top_k)
    
    @handle_exceptions
    def cmd_interpretability(self, *args):
        """
        Compute interpretability metrics.
        Usage: interpretability <metric_name> [<args>]
        """
        if not args:
            return "Usage: interpretability <metric_name> [<args>]"
        metric_name = args[0]
        metric_args = args[1:]
        
        metrics = InterpretabilityMetrics()
        if hasattr(metrics, metric_name):
            method = getattr(metrics, metric_name)
            return method(self.debugger.wrapped_model.model, *metric_args)
        else:
            return f"Unknown metric: {metric_name}"
    
    @handle_exceptions
    def cmd_explainability(self, *args):
        """
        Apply explainability techniques.
        Usage: explainability <technique_name> [<args>]
        """
        if not args:
            return "Usage: explainability <technique_name> [<args>]"
        technique_name = args[0]
        technique_args = args[1:]
        
        techniques = ExplainabilityTechniques()
        if hasattr(techniques, technique_name):
            method = getattr(techniques, technique_name)
            return method(self.debugger.wrapped_model.model, *technique_args)
        else:
            return f"Unknown technique: {technique_name}"
        
    @handle_exceptions
    def cmd_add_hook(self, *args):
        """
        Add a custom hook to a module.
        Usage: add_hook <forward|backward> <module_name> <hook_name> <hook_function>
        """
        if len(args) != 4:
            return "Usage: add_hook <forward|backward> <module_name> <hook_name> <hook_function>"
        
        hook_type, module_name, hook_name, hook_function = args
        
        try:
            hook = eval(f"lambda module, input, output: {hook_function}")
        except Exception as e:
            return f"Error in hook function: {str(e)}"
        
        if hook_type == "forward":
            return self.debugger.custom_hook_manager.register_forward_hook(module_name, hook, hook_name)
        elif hook_type == "backward":
            return self.debugger.custom_hook_manager.register_backward_hook(module_name, hook, hook_name)
        else:
            return "Invalid hook type. Use 'forward' or 'backward'."

    @handle_exceptions
    def cmd_remove_hook(self, *args):
        """
        Remove a custom hook.
        Usage: remove_hook <hook_name>
        """
        if len(args) != 1:
            return "Usage: remove_hook <hook_name>"
        
        hook_name = args[0]
        return self.debugger.custom_hook_manager.remove_hook(hook_name)

    @handle_exceptions
    def cmd_list_hooks(self, *args):
        """
        List all registered custom hooks.
        Usage: list_hooks
        """
        return self.debugger.custom_hook_manager.list_hooks()

    @handle_exceptions
    def cmd_clear_hooks(self, *args):
        """
        Clear all custom hooks.
        Usage: clear_hooks
        """
        return self.debugger.custom_hook_manager.clear_all_hooks()