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
        P   erform analysis on the model.
        Usage: analyze <type> [<args>]
        Types: gradients, attention, activations, tokens
        """
        if not args:
            return "Usage: analyze <type> [<args>]"
        analysis_type = args[0]
        if analysis_type == "gradients":
            return self.debugger.gradient_inspector.inspect(args[1] if len(args) > 1 else None)
        elif analysis_type == "attention":
            return self.debugger.attention_inspector.inspect(args[1] if len(args) > 1 else None)
        elif analysis_type == "activations":
            return self.debugger.activation_inspector.inspect(args[1] if len(args) > 1 else None)
        elif analysis_type == "tokens":
            input_text = " ".join(args[1:-1]) if len(args) > 2 and args[-1].isdigit() else " ".join(args[1:])
            top_k = int(args[-1]) if len(args) > 2 and args[-1].isdigit() else 5
            return self.debugger.analyze_tokens(input_text, top_k)
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
    def cmd_reset_weights(self, *args):
        """
        Reset all modified weights to their original values.
        Usage: reset_weights
        """
        return self.debugger.reset_modified_weights()

    @handle_exceptions
    def cmd_analyze_tokens(self, *args):
        """
        Analyze tokens with various methods.
        Usage: analyze_tokens <input_text> <analysis_type> [options]
        Analysis types: probabilities, saliency, attention, counterfactual, attribution, neuron_activation, representation_tracking, clustering, importance_ranking
        Options:
          --top_k <int>: Number of top tokens to show (for probabilities)
          --compare_modified: Compare with modified weights (for probabilities)
          --layer <int>: Layer to analyze (for attention, neuron_activation, clustering)
          --head <int>: Attention head to analyze (for attention)
          --token_index <int>: Token to analyze (for counterfactual)
          --replacements <str1,str2,...>: Replacement tokens (for counterfactual)
          --method <str>: Attribution method (for attribution)
          --n_clusters <int>: Number of clusters (for clustering)
        """
        if len(args) < 2:
            return self.cmd_analyze_tokens.__doc__

        # Find the index of the analysis type
        analysis_type_index = next((i for i, arg in enumerate(args) if arg in ['probabilities', 'saliency', 'attention', 'counterfactual', 'attribution', 'neuron_activation', 'representation_tracking', 'clustering', 'importance_ranking']), None)

        if analysis_type_index is None:
            return "Invalid analysis type. Please specify a valid analysis type."

        input_text = " ".join(args[:analysis_type_index])
        analysis_type = args[analysis_type_index]
        options = self._parse_options(args[analysis_type_index+1:])

        # Convert layer and head to integers if present
        if 'layer' in options:
            options['layer'] = int(options['layer'])
        if 'head' in options:
            options['head'] = int(options['head'])

        compare_modified = options.pop('compare_modified', False)
        return self.debugger.analyze_tokens(input_text, analysis_type, compare_modified, **options)


    @handle_exceptions
    def cmd_analyze_token_attention_representation(self, *args):
        """
        Analyze token attention and/or representation.
        Usage: analyze_token_attention_representation <input_text> [options]
        Options:
          --layer <int>: Layer to analyze (default: -1)
          --head <int>: Attention head to analyze (default: None, means all heads)
          --attention <bool>: Include attention analysis (default: True)
          --representation <bool>: Include representation analysis (default: True)
        """
        if len(args) < 1:
            return self.cmd_analyze_token_attention_representation.__doc__

        input_text = args[0]
        options = self._parse_options(args[1:])

        return self.debugger.analyze_token_attention_and_representation(input_text, **options)

    
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
    def cmd_hook(self, *args):
        """
        Manage hooks in the model.
        Usage:
            <nngdb {hook}> add <forward|backward> <module_name> <hook_name> <hook_function>
            <nngdb {hook}> remove <hook_name>
            <nngdb {hook}> list
            <nngdb {hook}> clear
        """
        if not args:
            return self.cmd_hook.__doc__

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
            return f"Unknown hook subcommand: {subcommand}"

    def _hook_add(self, args):
        if len(args) < 4:
            return "Usage: <nngdb {hook}> add <forward|backward> <module_name> <hook_name> <hook_function>"
        hook_type, module_name, hook_name = args[0], args[1], args[2]
        hook_function = " ".join(args[3:])
        return self.debugger.add_hook(hook_type, module_name, hook_name, hook_function)

    def _hook_remove(self, args):
        if not args:
            return "Usage: <nngdb {hook}> remove <hook_name>"
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
            return self.cmd_experiment.__doc__

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
            return f"Unknown experiment subcommand: {subcommand}"

    def _experiment_create(self, args):
        """
        Create a new experiment.
        Usage: experiment create <name>
        """
        if len(args) != 1:
            return "Usage: experiment create <name>"
        return self.debugger.create_experiment(args[0])

    def _experiment_switch(self, args):
        """
        Switch to an existing experiment.
        Usage: experiment switch <name>
        """
        if len(args) != 1:
            return "Usage: experiment switch <name>"
        return self.debugger.switch_experiment(args[0])

    def _experiment_list(self, args):
        """
        List all available experiments.
        Usage: experiment list
        """
        experiments = self.debugger.list_experiments()
        if not experiments:
            return "No experiments available."
        return "Available experiments:\n" + "\n".join(experiments)

    def _experiment_delete(self, args):
        """
        Delete an existing experiment.
        Usage: experiment delete <name>
        """
        if len(args) != 1:
            return "Usage: experiment delete <name>"
        return self.debugger.delete_experiment(args[0])

    def _experiment_compare(self, args):
        """
        Compare two experiments.
        Usage: experiment compare <exp1> <exp2> <input_text> [analysis_type] [options]
        """
        if len(args) < 3:
            return "Usage: experiment compare <exp1> <exp2> <input_text> [analysis_type] [options]"
        exp1, exp2 = args[0], args[1]
        input_text = args[2]
        analysis_type = args[3] if len(args) > 3 else 'probabilities'
        options = self._parse_options(args[4:])
        return self.debugger.compare_experiments(exp1, exp2, input_text, analysis_type, **options)

    def _experiment_current(self, args):
        """
        Show the current active experiment.
        Usage: experiment current
        """
        current_exp = self.debugger.get_current_experiment()
        return f"Current experiment: {current_exp}" if current_exp else "No active experiment."
