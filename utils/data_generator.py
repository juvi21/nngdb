import torch
from transformers import PreTrainedTokenizer

def generate_random_input(tokenizer: PreTrainedTokenizer, max_length: int = 50):
    vocab_size = tokenizer.vocab_size
    random_ids = torch.randint(0, vocab_size, (1, max_length))
    return random_ids

def generate_adversarial_input(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, target_label: int, max_iterations: int = 100):
    input_ids = generate_random_input(tokenizer)
    input_ids.requires_grad = True
    optimizer = torch.optim.Adam([input_ids], lr=0.1)
    
    for _ in range(max_iterations):
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = -outputs.logits[0, target_label]
        loss.backward()
        optimizer.step()
        
        input_ids.data = torch.clamp(input_ids.data, 0, tokenizer.vocab_size - 1)
    
    return input_ids.detach().long()