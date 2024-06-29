class TokenGradientTracker:
    def __init__(self, model_wrapper, tokenizer):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer

    def track_token_gradient(self, token, layers=None):
        token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
        
        if layers is None:
            layers = list(self.model_wrapper.current_state.keys())
        
        result = f"Gradient tracking for token '{token}':\n"
        for layer in layers:
            layer_state = self.model_wrapper.current_state.get(layer, {})
            grad_input = layer_state.get('grad_input')
            if grad_input is None or len(grad_input) == 0:
                result += f"{layer}: No gradient information available\n"
                continue
            
            token_grad = grad_input[0][:, token_id]
            result += f"{layer}:\n"
            result += f"  Gradient norm: {token_grad.norm().item():.4f}\n"
            result += f"  Gradient mean: {token_grad.mean().item():.4f}\n"
            result += f"  Gradient std: {token_grad.std().item():.4f}\n"
        
        return result