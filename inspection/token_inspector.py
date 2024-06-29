import torch

class TokenInspector:
    def __init__(self, model_wrapper, tokenizer):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer

    def get_token_activation(self, layer_name, token, model_output):
        token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
        layer_output = self.model_wrapper.current_state.get(layer_name, {}).get('output')
        if layer_output is None:
            return f"No activation data available for layer '{layer_name}'"
        if layer_output.dim() < 2 or layer_output.size(1) <= token_id:
            return f"Invalid token ID or layer output shape"
        return layer_output[:, token_id]

    def get_position_activation(self, layer_name, position):
        layer_output = self.model_wrapper.current_state.get(layer_name, {}).get('output')
        if layer_output is None:
            return f"No activation data available for layer '{layer_name}'"
        if layer_output.dim() < 2 or layer_output.size(1) <= position:
            return f"Invalid position or layer output shape"
        return layer_output[:, position]

    def get_token_gradient(self, layer_name, token):
        token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
        layer_state = self.model_wrapper.current_state.get(layer_name, {})
        grad_input = layer_state.get('grad_input')
        if grad_input is None or len(grad_input) == 0:
            return f"No gradient information available for layer '{layer_name}'"
        return grad_input[0][:, token_id]