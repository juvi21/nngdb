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