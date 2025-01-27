import torch
import heapq
from typing import Dict, List, Tuple
import traceback


class DatasetExampleCollector:
    def __init__(self, num_top_examples: int = 10):
        self.num_top_examples = num_top_examples
        self.layer_activations = {}
        self.token_activations = {}

    def collect_activations(self, layer_name: str, activations: torch.Tensor, input_tokens: List[str]):
        if layer_name not in self.layer_activations:
            self.layer_activations[layer_name] = [[] for _ in range(activations.size(-1))]
        
        for neuron_idx in range(activations.size(-1)):
            neuron_activations = activations[:, neuron_idx]
            top_activations = self.layer_activations[layer_name][neuron_idx]

            for token_idx, activation in enumerate(neuron_activations):
                if token_idx < len(input_tokens):
                    activation_value = activation.item()
                    token = input_tokens[token_idx]

                    if token not in self.token_activations:
                        self.token_activations[token] = {}
                    if layer_name not in self.token_activations[token]:
                        self.token_activations[token][layer_name] = []

                    self.token_activations[token][layer_name].append((activation_value, neuron_idx))

                    if len(top_activations) < self.num_top_examples:
                        heapq.heappush(top_activations, (activation_value, token))
                    elif activation_value > top_activations[0][0]:
                        heapq.heapreplace(top_activations, (activation_value, token))

    def get_top_examples(self, layer_name: str) -> List[List[Tuple[float, str]]]:
        if layer_name not in self.layer_activations:
            return []
        return [sorted(neuron_activations, reverse=True) for neuron_activations in self.layer_activations[layer_name]]

    def get_token_activations(self, token: str) -> Dict[str, List[Tuple[float, int]]]:
        return self.token_activations.get(token, {})

    def clear(self):
        self.layer_activations.clear()
        self.token_activations.clear()