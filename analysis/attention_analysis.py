import torch
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_attention_weights(attention_weights: torch.Tensor, tokens: list):
    att_mat = attention_weights.squeeze().cpu().detach().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(att_mat, annot=True, cmap='viridis', ax=ax)
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens, rotation=0)
    ax.set_title('Attention Weights')
    plt.tight_layout()
    plt.show()

def analyze_attention_patterns(model, input_ids: torch.Tensor):
    attention_patterns = []
    for layer in model.transformer.h:
        attention_output = layer.attn(input_ids)[0]
        attention_patterns.append(attention_output)
    return attention_patterns