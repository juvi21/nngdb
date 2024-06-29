class TokenAttentionVisualizer:
    def __init__(self, model_wrapper, tokenizer):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer

    def visualize_token_attention(self, token, layer=None):
        token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
        
        if layer is None:
            attention_layers = [l for l in self.model_wrapper.current_state.keys() if 'attention' in l.lower()]
            if not attention_layers:
                return "No attention layers found"
            layer = attention_layers[-1]
        
        attention_weights = self.model_wrapper.current_state.get(layer, {}).get('output')
        if attention_weights is None:
            return f"No attention weights available for layer '{layer}'"
        
        if attention_weights.dim() == 3:
            token_attention = attention_weights[0, :, token_id]
        elif attention_weights.dim() == 4:
            token_attention = attention_weights[0, :, :, token_id].mean(dim=0)
        else:
            return f"Unexpected attention weights shape: {attention_weights.shape}"
        
        result = f"Attention for token '{token}' in layer '{layer}':\n"
        for i, weight in enumerate(token_attention):
            result += f"Position {i}: {'#' * int(weight * 50)} ({weight:.4f})\n"
        
        return result