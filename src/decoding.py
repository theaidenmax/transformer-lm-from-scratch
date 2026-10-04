import torch
from src.model import softmax

def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
     if p >= 1.0:
          return probs
     sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

     sorted_indices_to_remove = cumulative_probs > p
     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
     sorted_indices_to_remove[..., 0] = 0

     sorted_probs[sorted_indices_to_remove] = 0.0
    
     sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

     probs_filtered = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
     return probs_filtered

@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int = None,
) -> torch.Tensor:
    model.eval()
    input_ids = prompt_ids.clone()

    for _ in range(max_new_tokens):
        cond_ids = input_ids[:, -context_length:]

        logits = model(cond_ids)
        
        next_token_logits = logits[:, -1, :]

        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            probs = softmax(next_token_logits, dim=-1)

            if top_p < 1.0:
                probs = sample_top_p(probs, top_p)

            next_token = torch.multinomial(probs, num_samples=1)

        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        input_ids = torch.cat((input_ids, next_token), dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return input_ids