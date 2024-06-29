import torch

class TokenActivationComparator:
    def __init__(self, model_wrapper, tokenizer):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer

    def compare_token_activations(self, token1, token2, layers=None):
        token_id1 = self.tokenizer.encode(token1, add_special_tokens=False)[0]
        token_id2 = self.tokenizer.encode(token2, add_special_tokens=False)[0]
        
        if layers is None:
            layers = list(self.model_wrapper.current_state.keys())
        
        result = f"Activation comparison for tokens '{token1}' and '{token2}':\n"
        for layer in layers:
            layer_output = self.model_wrapper.current_state.get(layer, {}).get('output')
            if layer_output is None:
                result += f"{layer}: No activation data available\n"
                continue
            
            if layer_output.dim() < 2 or layer_output.size(1) <= max(token_id1, token_id2):
                result += f"{layer}: Invalid token ID or layer output shape\n"
                continue
            
            act1 = layer_output[:, token_id1]
            act2 = layer_output[:, token_id2]
            
            similarity = torch.nn.functional.cosine_similarity(act1, act2, dim=0)
            result += f"{layer}: Cosine similarity = {similarity.item():.4f}\n"
        
        return result