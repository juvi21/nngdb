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
            'inspect': "Inspect various aspects of the model (model, layer, weight, activation, gradient).",
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
        Inspect various aspects of the model.
        Usage: 
            inspect model
            inspect layer <layer_name>
            inspect weight <layer_name> <weight_name>
            inspect activation <layer_name> [--token <token> | --position <pos>]
            inspect gradient <layer_name> [--token <token>]
        """
        if len(args) < 1:
            return "Error: Missing arguments. Use 'help inspect' for usage information."

        inspect_type = args[0]
        if inspect_type == "model":
            return self.debugger.inspect_model()
        elif inspect_type in ["layer", "weight", "activation", "gradient"]:
            if len(args) < 2:
                return f"Error: Missing layer name for {inspect_type} inspection."
            layer_name = args[1]
            if inspect_type == "layer":
                return self.debugger.inspect_layer(layer_name)
            elif inspect_type == "weight":
                if len(args) < 3:
                    return "Error: Missing weight name. Usage: inspect weight <layer_name> <weight_name>"
                weight_name = args[2]
                return self.debugger.inspect_weights(layer_name, weight_name)
            elif inspect_type == "activation":
                options = self._parse_options(args[2:])
                if 'token' in options:
                    return self.debugger.get_token_activation(layer_name, options['token'])
                elif 'position' in options:
                    return self.debugger.get_position_activation(layer_name, int(options['position']))
                else:
                    return self.debugger.inspect_activations(layer_name)
            elif inspect_type == "gradient":
                options = self._parse_options(args[2:])
                if 'token' in options:
                    return self.debugger.get_token_gradient(layer_name, options['token'])
                else:
                    return self.debugger.inspect_gradients(layer_name)
        else:
            return f"Unknown inspection type: {inspect_type}. Use 'help inspect' for usage information."
    
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
            analyze attention <token> [<layer_name>]
            analyze neuron <token> <layer_name> [--top <n>]
            analyze gradient <token> [<layer1> <layer2> ...]
            analyze compare <token1> <token2> [<layer1> <layer2> ...]
            analyze probability <token1> [<token2> ...] [--input <text>]
        """
        if len(args) < 2:
            return "Error: Insufficient arguments. Use 'help analyze' for usage information."

        analysis_type = args[0]
        if analysis_type == "tokens":
            return self._analyze_tokens(args[1:])
        elif analysis_type == "attention":
            if len(args) < 2:
                return "Error: Missing token. Usage: analyze attention <token> [<layer_name>]"
            token = args[1]
            layer = args[2] if len(args) > 2 else None
            return self.debugger.visualize_token_attention(token, layer)
        elif analysis_type == "neuron":
            if len(args) < 3:
                return "Error: Insufficient arguments. Usage: analyze neuron <token> <layer_name> [--top <n>]"
            token, layer_name = args[1], args[2]
            options = self._parse_options(args[3:])
            top_n = int(options.get('top', 10))
            return self.debugger.analyze_token_neuron_activation(token, layer_name, top_n)
        elif analysis_type == "gradient":
            if len(args) < 2:
                return "Error: Missing token. Usage: analyze gradient <token> [<layer1> <layer2> ...]"
            token = args[1]
            layers = args[2:] if len(args) > 2 else None
            return self.debugger.track_token_gradient(token, layers)
        elif analysis_type == "compare":
            if len(args) < 3:
                return "Error: Insufficient arguments. Usage: analyze compare <token1> <token2> [<layer1> <layer2> ...]"
            token1, token2 = args[1], args[2]
            layers = args[3:] if len(args) > 3 else None
            return self.debugger.compare_token_activations(token1, token2, layers)
        elif analysis_type == "probability":
            return self._analyze_token_probabilities(args[1:])
        else:
            return f"Unknown analysis type: {analysis_type}. Use 'help analyze' for usage information."

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
            trace start
            trace stop
            trace get <type> [<layer_name>]
            trace clear
            trace follow <token>
        Types: execution, activations, gradients
        """
        if not args:
            return "Error: No trace action specified. Usage: trace <subcommand> <args>"

        subcommand = args[0]
        if subcommand == "start":
            self.debugger.tracing_enabled = True
            return "Tracing enabled. Run the model to collect traces."
        elif subcommand == "stop":
            self.debugger.tracing_enabled = False
            return "Tracing disabled."
        elif subcommand == "get":
            return self._trace_get(args[1:])
        elif subcommand == "clear":
            return self._trace_clear()
        elif subcommand == "follow":
            return self._trace_follow(args[1:])
        else:
            return f"Unknown trace subcommand: {subcommand}. Valid subcommands are: start, stop, get, clear, follow."

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
    
    def _trace_follow(self, args):
        if not args:
            return "Error: No token specified. Usage: trace follow <token>"
        token = args[0]
        token_ids = self.debugger.tokenizer.encode(token, add_special_tokens=False)
        if not token_ids:
            return f"Error: '{token}' could not be tokenized."
    
        result = f"Trace for token sequence '{token}':\n"
        for i, token_id in enumerate(token_ids):
            sub_token = self.debugger.tokenizer.decode([token_id])
            token_trace = self.debugger.get_token_trace(sub_token)
            if not token_trace:
                result += f"No trace found for sub-token '{sub_token}' (ID: {token_id}) at position {i}.\n"
                continue
        
            result += f"Sub-token: '{sub_token}' (ID: {token_id}) at position {i}:\n"
            for step in token_trace:
                result += f"  Layer: {step['layer_name']}\n"
                result += f"  Output shape: {step['output'].shape}\n"
                result += f"  Output mean: {step['output'].mean().item():.4f}\n"
                result += f"  Output std: {step['output'].std().item():.4f}\n\n"
    
        return result
    
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
            elif args[i] == '--top' and i + 1 < len(args):
                try:
                    top_n = int(args[i+1])
                    i += 1
                except ValueError:
                    return f"Error: Invalid value for --top: {args[i+1]}"
            elif args[i] == '--neuron' and i + 1 < len(args):
                try:
                    neuron_idx = int(args[i+1])
                    i += 1
                except ValueError:
                    return f"Error: Invalid value for --neuron: {args[i+1]}"
            elif input_source is None and not inline_mode:
                input_source = args[i]
            else:
                layer_names.append(args[i])
            i += 1

        if not layer_names:
            return "Error: No layer names specified."

        if inline_mode:
            input_texts = self._get_inline_dataset()
        elif input_source:
            try:
                with open(input_source, 'r') as f:
                    input_texts = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                return f"Error: Input file '{input_source}' not found."
            except IOError:
                return f"Error: Unable to read file '{input_source}'."
        else:
            return "Error: No input source specified. Use --inline or provide an input file."

        if not input_texts:
            return "Error: No input texts provided."

        result = self.debugger.collect_dataset_examples(input_texts, layer_names, top_n)

        if not result:
            return "Error: No results returned from collect_dataset_examples."

        if neuron_idx is not None:
            return self._format_neuron_examples(result, layer_names, neuron_idx)
        elif '--token' in args:
            token_index = args.index('--token')
            if token_index + 1 < len(args):
                token = args[token_index + 1]
                return self._format_token_examples(result, token)
            else:
                return "Error: No token provided after --token flag."
        else:
            return self._format_layer_examples(result, layer_names)
    
    @handle_exceptions
    def _format_token_examples(self, result, token):
        token_activations = self.debugger.dataset_example_collector.get_token_activations(token)
        if not token_activations:
            return f"No activations found for token '{token}'"

        formatted = f"Activations for token '{token}':\n"
        for layer, activations in token_activations.items():
            formatted += f"Layer: {layer}\n"
            sorted_activations = sorted(activations, reverse=True)[:self.debugger.dataset_example_collector.num_top_examples]
            for activation, neuron_idx in sorted_activations:
                formatted += f"  Neuron {neuron_idx}: {activation:.4f}\n"
            formatted += "\n"
        return formatted

    @handle_exceptions
    def _get_inline_dataset(self):
        print("Enter your dataset examples. Type 'END' on a new line when finished:")
        examples = []
        while True:
            try:
                line = input().strip()
                if line.upper() == 'END':
                    break
                if line:  # Only add non-empty lines
                    examples.append(line)
            except EOFError:
                break  # Handle Ctrl+D (EOF)
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
            if layer in result and result[layer]:
                formatted += f"Top activated neurons for layer '{layer}':\n"
                for neuron_idx, neuron_activations in enumerate(result[layer]):
                    if neuron_activations:
                        top_activation, top_token = neuron_activations[0]
                        formatted += f"  Neuron {neuron_idx}: {top_token} ({top_activation:.4f})\n"
            else:
                formatted += f"No examples found for layer '{layer}'\n"
        return formatted
    
    @handle_exceptions
    def _analyze_token_probabilities(self, args):
        if len(args) < 1:
            return "Error: No tokens provided. Usage: analyze probability <token1> [<token2> ...] [--input <text>]"
    
        options = self._parse_options(args)
        tokens = [arg for arg in args if not arg.startswith('--')]
        input_text = options.get('input')
    
        return self.debugger.compare_token_probabilities(tokens, input_text)