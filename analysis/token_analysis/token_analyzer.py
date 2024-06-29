import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from transformers import AutoModel, AutoModelForCausalLM

#Experimental
class TokenAnalyzer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.history = []

    def analyze(self, input_text: str, analysis_type: str, compare_modified: bool = False, **kwargs):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        if analysis_type == 'probabilities':
            original_result = self._analyze_token_probabilities(input_ids, tokens, **kwargs)
            if not compare_modified:
                return original_result
            
            # If comparing with modified weights, perform the analysis again
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            probs = torch.softmax(logits[0, -1], dim=-1)
            top_k = kwargs.get('top_k', 5)
            top_probs, top_indices = torch.topk(probs, top_k)
            
            modified_result = {
                "input_text": input_text,
                "top_tokens": [
                    (self.tokenizer.decode([idx.item()]), prob.item())
                    for idx, prob in zip(top_indices, top_probs)
                ]
            }
            
            return {
                "original": original_result,
                "modified": modified_result
            }
        else:
            analysis_methods = {
                'saliency': self._token_saliency,
                'attention': self._visualize_attention,
                'counterfactual': self._counterfactual_analysis,
                'attribution': self._token_attribution,
                'neuron_activation': self._neuron_activation_by_token,
                'representation_tracking': self._track_token_representations,
                'clustering': self._cluster_tokens,
                'importance_ranking': self._rank_token_importance
            }

            if analysis_type not in analysis_methods:
                return f"Unknown analysis type: {analysis_type}"

            return analysis_methods[analysis_type](input_ids, tokens, **kwargs)

    def analyze_attention_and_representation(self, input_text: str, layer: int = -1, head: int = None, 
                                             include_attention: bool = True, include_representation: bool = True) -> Dict[str, Any]:
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        results = {}
        
        if include_attention:
            results['attention'] = self._visualize_attention(input_ids, tokens, layer=layer, head=head)
        
        if include_representation:
            results['representation'] = self._track_token_representations(input_ids, tokens)
        
        return results

    def _analyze_token_probabilities(self, input_ids: torch.Tensor, tokens: List[str], top_k: int = 5) -> Dict[str, Any]:
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        probs = F.softmax(logits[0, -1], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        return {
            "input_text": self.tokenizer.decode(input_ids[0]),
            "top_tokens": [
                (self.tokenizer.decode([idx.item()]), prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
        }

    def _token_saliency(self, input_ids, tokens):
        self.model.zero_grad()
        embed = self.model.get_input_embeddings()
        
        input_ids = input_ids.to(self.model.device)
        input_embed = embed(input_ids)
        input_embed.retain_grad()
        
        outputs = self.model(inputs_embeds=input_embed)
        output = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        output.sum().backward()
        
        saliency = input_embed.grad.abs().sum(dim=-1)
        
        return {
            "tokens": tokens,
            "saliency": saliency[0].tolist()
        }

    def _visualize_attention(self, input_ids, tokens, layer=None, head=None):
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Determine the number of layers
        if isinstance(self.model, AutoModelForCausalLM):
            num_layers = len(self.model.transformer.h)
        elif hasattr(self.model, 'config'):
            num_layers = self.model.config.num_hidden_layers
        else:
            return "Unable to determine the number of layers in the model."

        # Handle layer selection
        layer = num_layers - 1 if layer is None else int(layer)
        layer = layer if layer >= 0 else num_layers + layer

        if layer < 0 or layer >= num_layers:
            return f"Invalid layer index. Model has {num_layers} layers."

        # Prepare inputs
        input_ids = input_ids.to(self.model.device)

        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)

        # Extract attention weights
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions[layer]
        else:
            return "Model does not output attention weights. Make sure you're using a model that supports outputting attention."

        # Handle head selection
        if head is not None:
            head = int(head)
            if head < 0 or head >= attentions.size(1):
                return f"Invalid head index. Layer has {attentions.size(1)} attention heads."
            attentions = attentions[:, head, :, :]
        else:
            attentions = attentions.mean(dim=1)

        attention_data = attentions[0].cpu().numpy()

        return {
            "tokens": tokens,
            "attention_weights": attention_data.tolist()
        }

    def _counterfactual_analysis(self, input_ids: torch.Tensor, tokens: List[str], token_index: int, replacement_tokens: List[str]) -> Dict[str, Any]:
        original_output = self.model(input_ids).logits
        results = []
        
        for replacement in replacement_tokens:
            replacement_id = self.tokenizer.encode(replacement, add_special_tokens=False)[0]
            new_input_ids = input_ids.clone()
            new_input_ids[0, token_index] = replacement_id
            new_output = self.model(new_input_ids).logits
            
            diff = (new_output - original_output).abs().mean().item()
            results.append((replacement, diff))
        
        return {
            "original_token": tokens[token_index],
            "counterfactuals": results
        }

    def _token_attribution(self, input_ids: torch.Tensor, tokens: List[str], method: str = 'integrated_gradients') -> Dict[str, Any]:
        if method == 'integrated_gradients':
            attributions = self._integrated_gradients(input_ids)
        else:
            return {"error": f"Unsupported attribution method: {method}"}
        
        return {
            "tokens": tokens,
            "attributions": attributions[0].tolist()
        }

    def _integrated_gradients(self, input_ids: torch.Tensor, steps: int = 50) -> torch.Tensor:
        baseline = torch.zeros_like(input_ids)
        attributions = torch.zeros_like(input_ids, dtype=torch.float)
        
        for step in range(1, steps + 1):
            interpolated_input = baseline + (step / steps) * (input_ids - baseline)
            interpolated_input.requires_grad_(True)
            
            outputs = self.model(interpolated_input)
            output = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            output.backward(torch.ones_like(output))
            
            attributions += interpolated_input.grad
        
        attributions /= steps
        return attributions * (input_ids - baseline)

    def _neuron_activation_by_token(self, input_ids: torch.Tensor, tokens: List[str], layer: int) -> Dict[str, Any]:
        def hook_fn(module, input, output):
            self.layer_output = output.detach()
        
        layer_module = list(self.model.modules())[layer]
        handle = layer_module.register_forward_hook(hook_fn)
        
        self.model(input_ids)
        handle.remove()
        
        activations = self.layer_output[0]
        top_neurons = activations.max(dim=-1).indices
        
        return {
            "tokens": tokens,
            "top_neurons": top_neurons.tolist()
        }

    def _track_token_representations(self, input_ids: torch.Tensor, tokens: List[str]) -> Dict[str, Any]:
        representations = []
        
        def hook_fn(module, input, output):
            representations.append(output[0].detach())
        
        handles = []
        for layer in self.model.encoder.layer:
            handles.append(layer.register_forward_hook(hook_fn))
        
        self.model(input_ids)
        
        for handle in handles:
            handle.remove()
        
        return {
            "tokens": tokens,
            "representations": [rep.cpu().numpy().tolist() for rep in representations]
        }

    def _cluster_tokens(self, input_ids: torch.Tensor, tokens: List[str], layer: int, n_clusters: int = 5) -> Dict[str, Any]:
        def hook_fn(module, input, output):
            self.layer_output = output[0].detach()
        
        layer_module = list(self.model.modules())[layer]
        handle = layer_module.register_forward_hook(hook_fn)
        
        self.model(input_ids)
        handle.remove()
        
        token_embeddings = self.layer_output.squeeze(0).cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(token_embeddings)
        
        return {
            "tokens": tokens,
            "clusters": clusters.tolist()
        }

    def _rank_token_importance(self, input_ids: torch.Tensor, tokens: List[str]) -> Dict[str, Any]:
        input_ids.requires_grad_(True)
        outputs = self.model(input_ids)
        output = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        output.backward(torch.ones_like(output))
        
        importance = input_ids.grad.abs().sum(dim=-1)
        sorted_indices = importance[0].argsort(descending=True)
        
        return {
            "tokens": tokens,
            "importance_ranking": sorted_indices.tolist()
        }