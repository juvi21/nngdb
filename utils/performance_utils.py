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