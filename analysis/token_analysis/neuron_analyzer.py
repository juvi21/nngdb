import torch

class TokenNeuronAnalyzer:
    def __init__(self, model_wrapper, tokenizer):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer

    def analyze_token_neuron_activation(self, token, layer, top_n=10):
        token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
        activation = self.model_wrapper.current_state.get(layer, {}).get('output')
        
        if activation is None:
            return f"No activation data available for layer '{layer}'"
        
        if activation.dim() < 2 or activation.size(1) <= token_id:
            return f"Invalid token ID or layer output shape"
        
        token_activation = activation[:, token_id]
        
        top_activations, top_indices = torch.topk(token_activation.squeeze(), top_n)
        
        result = f"Top {top_n} neuron activations for token '{token}' in layer '{layer}':\n"
        for i, (act, idx) in enumerate(zip(top_activations, top_indices)):
            result += f"{i+1}. Neuron {idx}: {act.item():.4f}\n"
        
        return result