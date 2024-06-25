import torch

class ExecutionEngine:
    def __init__(self, wrapped_model):
        self.wrapped_model = wrapped_model

    def run(self, input_data: torch.Tensor):
        self.wrapped_model.execution_paused = False
        self.wrapped_model.current_step = 0
        output = self.wrapped_model(input_data)
        if self.wrapped_model.execution_paused:
            print(f"Execution paused at {self.wrapped_model.current_layer}. Use 'continue' to resume or 'step' to go to the next layer.")
        return output

    def continue_execution(self):
        self.wrapped_model.execution_paused = False
        self.wrapped_model.step_mode = False
        print("Continuing execution...")

    def step(self, num_steps: int = 1):
        self.wrapped_model.execution_paused = False
        self.wrapped_model.step_mode = True
        current_layer_index = self.wrapped_model.layer_order.index(self.wrapped_model.current_layer)
        target_layer_index = min(current_layer_index + num_steps, len(self.wrapped_model.layer_order) - 1)
        target_layer = self.wrapped_model.layer_order[target_layer_index]
        print(f"Stepping {num_steps} layer(s) to {target_layer}...")