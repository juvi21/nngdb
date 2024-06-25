import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from debugger.core import NNGDB
from utils.visualizer import visualize_weights, visualize_activations
from utils.tensor_utils import tensor_statistics
from analysis.gradient_flow import analyze_gradient_flow, detect_vanishing_exploding_gradients
from analysis.attention_analysis import visualize_attention_weights, analyze_attention_patterns
from analysis.neuron_activation import get_top_k_activated_neurons, visualize_neuron_activations
from analysis.model_comparison import compare_model_outputs, find_diverging_layers
from experiment.tracker import ExperimentTracker
from experiment.config import load_config, save_config
from utils.model_surgery import freeze_layers, replace_layer
from utils.data_generator import generate_random_input, generate_adversarial_input

class CommandHandler:
    def __init__(self, debugger: NNGDB):
        self.debugger = debugger
        self.experiment_tracker = ExperimentTracker("NNGDB_Experiment")

    def handle_command(self, command: str, args: list) -> str:
        if command == "help":
            return self.show_help(args)
        elif command in ["b", "break"]:
            return self.set_breakpoint(args)
        elif command in ["c", "continue"]:
            return self.continue_execution()
        elif command in ["s", "step"]:
            return self.step(args)
        elif command in ["p", "print"]:
            return self.print_value(args)
        elif command == "run":
            return self.run(args)
        elif command == "info":
            return self.get_info(args)
        elif command in ["v", "visualize"]:
            return self.visualize(args)
        elif command == "analyze":
            return self.analyze(args)
        elif command == "compare":
            return self.compare_models(args)
        elif command == "experiment":
            return self.manage_experiment(args)
        elif command == "modify":
            return self.modify_model(args)
        elif command == "generate":
            return self.generate_data(args)
        elif command == "modify_weight":
            return self.modify_weight(args)
        elif command == "reset_model":
            return self.reset_model()
        elif command == "compare_activations":
            return self.compare_activations(args)
        else:
            return f"Unknown command: {command}"
    def show_help(self, args):
        if not args:
            return """
Available commands:
  help [command]   Show this help message or get help for a specific command
  break (b)        Set a breakpoint
  continue (c)     Continue execution after a breakpoint
  step (s)         Step through execution
  run              Run the model with given input
  print (p)        Print the value of an expression
  info             Get information about the model or debugger state
  visualize (v)    Visualize weights or activations
  analyze          Analyze gradient flow, attention, or activations
  compare          Compare model outputs or layers
  experiment       Manage experiment tracking
  modify           Modify the model (freeze or replace layers)
  generate         Generate random or adversarial input
  modify_weight    Modify a specific weight in the model
  reset_model      Reset the model to its initial state
  compare_activations  Compare activations before and after modifications

Type 'help <command>' for more information on a specific command.
"""
        else:
            command = args[0]
            if command in ["b", "break"]:
                return "Usage: break <layer_name> [<condition>]\nSet a breakpoint at the specified layer, optionally with a condition."
            elif command in ["c", "continue"]:
                return "Usage: continue\nContinue execution after hitting a breakpoint."
            elif command in ["s", "step"]:
                return "Usage: step [<num_steps>]\nStep through execution, optionally specifying the number of steps."
            elif command == "run":
                return "Usage: run <input_text>\nRun the model with the given input text."
            elif command in ["p", "print"]:
                return "Usage: print <expression>\nPrint the value of the given expression."
            elif command == "info":
                return "Usage: info <topic>\nGet information about the specified topic (e.g., 'breakpoints', 'layers')."
            elif command in ["v", "visualize"]:
                return "Usage: visualize <weights|activations> <layer_name>\nVisualize weights or activations for the specified layer."
            elif command == "analyze":
                return "Usage: analyze <gradient_flow|attention|activations> [<layer_name>]\nAnalyze gradient flow, attention weights, or neuron activations."
            elif command == "compare":
                return "Usage: compare <outputs|layers>\nCompare outputs or layers between two models."
            elif command == "experiment":
                return "Usage: experiment <start|log_param|log_metric|end> [<args>]\nManage experiment tracking."
            elif command == "modify":
                return "Usage: modify <freeze|replace> <args>\nModify the model by freezing layers or replacing a layer."
            elif command == "generate":
                return "Usage: generate <random|adversarial> [<args>]\nGenerate random or adversarial input."
            elif command == "modify_weight":
                return "Usage: modify_weight <layer_name> <weight_name> <indices> <value>\nModify a specific weight in the model."
            elif command == "reset_model":
                return "Usage: reset_model\nReset the model to its initial state."
            elif command == "compare_activations":
                return "Usage: compare_activations <layer_name>\nCompare activations before and after modifications for the specified layer."
            else:
                return f"Unknown command: {command}"

    def set_breakpoint(self, args):
        if len(args) < 1:
            return "Usage: break <layer_name> [<condition>]"
        layer_name = args[0]
        condition = " ".join(args[1:]) if len(args) > 1 else None
        return self.debugger.breakpoint_manager.set_breakpoint(layer_name, condition)

    def continue_execution(self):
        return self.debugger.execution_engine.continue_execution()

    def step(self, args):
        num_steps = int(args[0]) if args else 1
        return self.debugger.execution_engine.step(num_steps)

    def print_value(self, args):
        if len(args) < 1:
            return "Usage: print <expression>"
        expr = " ".join(args)
        try:
            result = eval(expr, {'self': self.debugger}, self.debugger.context)
            return f"{expr} = {result}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    from transformers.modeling_outputs import CausalLMOutputWithPast

    def run(self, args):
        input_text = " ".join(args)
        if not input_text:
            return "Usage: run <input_text>"
        self.debugger.set_context('input_text', input_text)
        tokenizer = self.debugger.get_context('tokenizer')
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        self.debugger.set_context('input_ids', input_ids)
    
        self.debugger.profiler.start()
        output = self.debugger.run(input_ids)
        self.debugger.profiler.stop()
    
        for layer_name, layer_state in self.debugger.wrapped_model.current_state.items():
            if 'output' in layer_state:
                if isinstance(layer_state['output'], torch.Tensor):
                    self.debugger.set_context(f'original_activations_{layer_name}', layer_state['output'].detach().clone())
                elif isinstance(layer_state['output'], tuple):
                    self.debugger.set_context(f'original_activations_{layer_name}', tuple(t.detach().clone() if isinstance(t, torch.Tensor) else t for t in layer_state['output']))
    
        if isinstance(output, torch.Tensor):
            logits = output
        elif isinstance(output, tuple):
            logits = output[0]  # Assuming the first element contains the logits
        elif isinstance(output, CausalLMOutputWithPast):
            logits = output.logits
        else:
            return f"Unexpected output type: {type(output)}"

        # Generate sequence
        generated_sequence = tokenizer.decode(logits[0].argmax(dim=-1), skip_special_tokens=True)
    
        # Generate next token prediction
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        next_token = tokenizer.decode([next_token_id])

        result = f"Input: {input_text}\n"
        result += f"Generated sequence: {generated_sequence}\n"
        result += f"Next token prediction: {next_token}\n"
        result += f"Execution time: {self.debugger.profiler.get_execution_time():.4f} seconds\n"
        result += "Run completed. Original activations stored for comparison."
        return result

    def get_info(self, args):
        if len(args) < 1:
            return "Usage: info <topic>"
        topic = args[0]
        if topic == "breakpoints":
            return self.debugger.breakpoint_manager.list_breakpoints()
        elif topic == "layers":
            return "\n".join(self.debugger.wrapped_model.layer_order)
        else:
            return f"Unknown info topic: {topic}"

    def visualize(self, args):
        if len(args) < 2:
            return "Usage: visualize <weights|activations> <layer_name>"
        viz_type, layer_name = args[0], args[1]
        if viz_type == "weights":
            weights = self.debugger.state_inspector.get_layer_parameters(layer_name)
            if weights:
                visualize_weights(weights)
                return "Weight visualization displayed"
            else:
                return f"No weights found for layer {layer_name}"
        elif viz_type == "activations":
            activations = self.debugger.state_inspector.inspect_neuron(layer_name)
            if activations is not None:
                visualize_activations(activations, layer_name)
                return "Activation visualization displayed"
            else:
                return f"No activations found for layer {layer_name}"
        else:
            return f"Unknown visualization type: {viz_type}"

    def analyze(self, args):
        if len(args) < 1:
            return "Usage: analyze <gradient_flow|attention|activations>"
        analysis_type = args[0]
        if analysis_type == "gradient_flow":
            analyze_gradient_flow(self.debugger.wrapped_model.model)
            vanishing, exploding = detect_vanishing_exploding_gradients(self.debugger.wrapped_model.model)
            return f"Gradient flow visualized. Vanishing gradients: {vanishing}, Exploding gradients: {exploding}"
        elif analysis_type == "attention":
            if len(args) < 2:
                return "Usage: analyze attention <layer_name>"
            layer_name = args[1]
            attention_weights = self.debugger.state_inspector.inspect_attention(layer_name)
            if attention_weights is not None:
                tokens = self.debugger.get_context('input_text').split()
                visualize_attention_weights(attention_weights, tokens)
                return "Attention weights visualized"
            else:
                return f"No attention weights found for layer {layer_name}"
        elif analysis_type == "activations":
            if len(args) < 2:
                return "Usage: analyze activations <layer_name>"
            layer_name = args[1]
            activations = self.debugger.state_inspector.inspect_neuron(layer_name)
            if activations is not None:
                visualize_neuron_activations(activations, layer_name)
                top_k_values, top_k_indices = get_top_k_activated_neurons(activations)
                return f"Neuron activations visualized. Top 10 activated neurons: {list(zip(top_k_indices.tolist(), top_k_values.tolist()))}"
            else:
                return f"No activations found for layer {layer_name}"
        else:
            return f"Unknown analysis type: {analysis_type}"

    def compare_models(self, args):
        if len(args) < 1:
            return "Usage: compare <outputs|layers>"
        compare_type = args[0]
        if compare_type == "outputs":
            model1 = self.debugger.wrapped_model.model
            model2 = self.debugger.get_context('comparison_model')
            if model2 is None:
                return "No comparison model set. Use 'set comparison_model <model_path>' first."
            input_ids = self.debugger.get_context('input_ids')
            outputs = compare_model_outputs([model1, model2], input_ids)
            diff = outputs['model_0'] - outputs['model_1']
            return f"Output difference: max={diff.max().item()}, mean={diff.mean().item()}"
        elif compare_type == "layers":
            model1 = self.debugger.wrapped_model.model
            model2 = self.debugger.get_context('comparison_model')
            if model2 is None:
                return "No comparison model set. Use 'set comparison_model <model_path>' first."
            input_ids = self.debugger.get_context('input_ids')
            diverging_layers = find_diverging_layers(model1, model2, input_ids)
            return f"Diverging layers: {diverging_layers}"
        else:
            return f"Unknown comparison type: {compare_type}"

    def manage_experiment(self, args):
        if len(args) < 1:
            return "Usage: experiment <start|log_param|log_metric|end>"
        action = args[0]
        if action == "start":
            if len(args) < 2:
                return "Usage: experiment start <run_name>"
            run_name = args[1]
            self.experiment_tracker.start_run(run_name)
            return f"Started experiment run: {run_name}"
        elif action == "log_param":
            if len(args) < 3:
                return "Usage: experiment log_param <key> <value>"
            key, value = args[1], args[2]
            self.experiment_tracker.log_params({key: value})
            return f"Logged parameter: {key}={value}"
        elif action == "log_metric":
            if len(args) < 3:
                return "Usage: experiment log_metric <key> <value>"
            key, value = args[1], float(args[2])
            self.experiment_tracker.log_metric(key, value)
            return f"Logged metric: {key}={value}"
        elif action == "end":
            self.experiment_tracker.end_run()
            return "Ended experiment run"
        else:
            return f"Unknown experiment action: {action}"

    def modify_model(self, args):
        if len(args) < 1:
            return "Usage: modify <freeze|replace>"
        action = args[0]
        if action == "freeze":
            if len(args) < 2:
                return "Usage: modify freeze <layer_names>"
            layers_to_freeze = args[1:]
            return freeze_layers(self.debugger.wrapped_model.model, layers_to_freeze)
        elif action == "replace":
            if len(args) < 3:
                return "Usage: modify replace <layer_name> <new_layer_type>"
            layer_name, new_layer_type = args[1], args[2]
            old_layer = eval(f"self.debugger.wrapped_model.model.{layer_name}")
            new_layer = eval(f"torch.nn.{new_layer_type}(*old_layer.args, **old_layer.kwargs)")
            replace_layer(self.debugger.wrapped_model.model, layer_name, new_layer)
            return f"Replaced layer {layer_name} with {new_layer_type}"
        else:
            return f"Unknown modification action: {action}"

    def generate_data(self, args):
        if len(args) < 1:
            return "Usage: generate <random|adversarial>"
        gen_type = args[0]
        tokenizer = self.debugger.get_context('tokenizer')
        if tokenizer is None:
            return "No tokenizer set. Use 'set tokenizer <tokenizer_path>' first."
        if gen_type == "random":
            max_length = int(args[1]) if len(args) > 1 else 50
            input_ids = generate_random_input(tokenizer, max_length)
            self.debugger.set_context('input_ids', input_ids)
            return f"Generated random input with length {max_length}"
        elif gen_type == "adversarial":
            if len(args) < 2:
                return "Usage: generate adversarial <target_label>"
            target_label = int(args[1])
            input_ids = generate_adversarial_input(self.debugger.wrapped_model.model, tokenizer, target_label)
            self.debugger.set_context('input_ids', input_ids)
            return f"Generated adversarial input for target label {target_label}"
        else:
            return f"Unknown generation type: {gen_type}"

    def modify_weight(self, args):
        if len(args) < 4:
            return "Usage: modify_weight <layer_name> <weight_name> <indices> <value>"
        layer_name, weight_name = args[0], args[1]
        indices = eval(args[2])  # Be cautious with eval, ensure input is sanitized
        value = float(args[3])
        self.debugger.state_modifier.modify_weight(layer_name, weight_name, indices, value)
        return f"Weight modified: {layer_name}.{weight_name}{indices} = {value}"

    def reset_model(self):
        self.debugger.state_modifier.reset_model()
        return "Model weights reset to initial values"

    def compare_activations(self, args):
        if len(args) < 1:
            return "Usage: compare_activations <layer_name>"
        layer_name = args[0]
        original_activations = self.debugger.get_context(f'original_activations_{layer_name}')
        if original_activations is None:
            return f"No original activations stored for {layer_name}. Run the model first and store activations."
    
        input_ids = self.debugger.get_context('input_ids')
        self.debugger.run(input_ids)
    
        current_activations = self.debugger.wrapped_model.current_state[layer_name]['output']
    
        def compare_tensors(orig, curr, name):
            if isinstance(orig, torch.Tensor) and isinstance(curr, torch.Tensor):
                diff = torch.abs(curr - orig)
                visualize_neuron_activations(diff, f"{name}_diff")
                max_diff = torch.max(diff).item()
                avg_diff = torch.mean(diff).item()
                return f"Max difference: {max_diff}, Average difference: {avg_diff}"
            return "Incompatible types for comparison"

        if isinstance(original_activations, torch.Tensor) and isinstance(current_activations, torch.Tensor):
            result = compare_tensors(original_activations, current_activations, layer_name)
        elif isinstance(original_activations, tuple) and isinstance(current_activations, tuple):
            results = [compare_tensors(orig, curr, f"{layer_name}_part{i}") 
                        for i, (orig, curr) in enumerate(zip(original_activations, current_activations))]
            result = "\n".join(results)
        elif isinstance(original_activations, CausalLMOutputWithPast) and isinstance(current_activations, CausalLMOutputWithPast):
            results = []
            for attr in ['last_hidden_state', 'past_key_values', 'hidden_states', 'attentions']:
                if hasattr(original_activations, attr) and hasattr(current_activations, attr):
                    orig_attr = getattr(original_activations, attr)
                    curr_attr = getattr(current_activations, attr)
                    if isinstance(orig_attr, torch.Tensor) and isinstance(curr_attr, torch.Tensor):
                        results.append(f"{attr}: " + compare_tensors(orig_attr, curr_attr, f"{layer_name}_{attr}"))
            result = "\n".join(results)
        else:
            return f"Incompatible activation types for comparison in layer {layer_name}"

        return f"Activation differences for {layer_name}:\n{result}\nVisualizations displayed."