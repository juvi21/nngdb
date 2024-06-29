import torch
from core.model_wrapper import ModelWrapper

class AttentionAnalyzer:
    def __init__(self, wrapped_model: ModelWrapper):
        self.wrapped_model = wrapped_model

    def analyze_attention_patterns(self, layer_name: str):
        if layer_name not in self.wrapped_model.current_state:
            return f"No attention data available for layer '{layer_name}'."

        layer_output = self.wrapped_model.current_state[layer_name]['output']
        
        if isinstance(layer_output, tuple):
            attention_weights = layer_output[1]
        elif isinstance(layer_output, torch.Tensor):
            attention_weights = layer_output
        else:
            return f"Unexpected output type for layer '{layer_name}': {type(layer_output)}"

        return self._analyze_attention(attention_weights)

    def _analyze_attention(self, attention_weights: torch.Tensor):
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        avg_attention = attention_weights.mean(dim=(0, 1))  # Average over batch and heads
        
        analysis = {
            "shape": attention_weights.shape,
            "num_heads": num_heads,
            "sequence_length": seq_len,
            "entropy": self._compute_attention_entropy(attention_weights),
            "top_k_attention": self._get_top_k_attention(avg_attention, k=5),
            "attention_to_self": self._compute_attention_to_self(avg_attention),
            "attention_to_neighbors": self._compute_attention_to_neighbors(avg_attention),
        }

        return analysis

    def _compute_attention_entropy(self, attention_weights: torch.Tensor):
        attention_probs = attention_weights.mean(dim=1)  # Average over heads
        entropy = -(attention_probs * torch.log(attention_probs + 1e-9)).sum(dim=-1).mean().item()
        return entropy

    def _get_top_k_attention(self, avg_attention: torch.Tensor, k: int):
        top_k_values, top_k_indices = torch.topk(avg_attention.mean(dim=0), k)
        return [(idx.item(), val.item()) for idx, val in zip(top_k_indices, top_k_values)]

    def _compute_attention_to_self(self, avg_attention: torch.Tensor):
        return torch.diag(avg_attention).mean().item()

    def _compute_attention_to_neighbors(self, avg_attention: torch.Tensor):
        seq_len = avg_attention.shape[0]
        neighbor_attention = torch.diag(avg_attention, diagonal=1) + torch.diag(avg_attention, diagonal=-1)
        return neighbor_attention.sum().item() / (2 * (seq_len - 1))

    def visualize_attention(self, layer_name: str):
        return "Attention visualization not implemented in this version."