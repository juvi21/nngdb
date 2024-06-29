import torch

class TokenProbabilityAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.history = []

    def analyze(self, input_text, top_k=5):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        probs = torch.softmax(logits[0, -1], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        result = {
            "input_text": input_text,
            "top_tokens": [
                (self.tokenizer.decode([idx.item()]), prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
        }
        self.history.append(result)
        return result

    def compare(self, index1, index2):
        if index1 >= len(self.history) or index2 >= len(self.history):
            return f"Invalid indices for comparison. Available range: 0-{len(self.history)-1}"
        
        result1 = self.history[index1]
        result2 = self.history[index2]
        
        comparison = f"Comparison:\n"
        comparison += f"Input 1: {result1['input_text']}\n"
        comparison += f"Input 2: {result2['input_text']}\n\n"
        comparison += "Top tokens:\n"
        
        for (token1, prob1), (token2, prob2) in zip(result1['top_tokens'], result2['top_tokens']):
            comparison += f"{token1} ({prob1:.4f}) vs {token2} ({prob2:.4f})\n"
        
        return comparison