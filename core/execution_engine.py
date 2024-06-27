import torch
from .model_wrapper import ModelWrapper

class ExecutionEngine:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model
        self.current_layer_index = 0

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