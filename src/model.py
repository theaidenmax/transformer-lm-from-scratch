import torch
import torch.nn as nn
import math
from einops import einsum
from torch.optim import Optimizer
from typing import Iterable

class Linear(nn.Module):

    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        weight_tensor = torch.empty(
            (out_features, in_features), device=device, dtype=dtype
        )
        self.weight = nn.Parameter(weight_tensor)

        self.device = device
        self.dtype = dtype

        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(
            self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")



class Embedding(nn.Module):

    def __init__(self, num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        weight_tensor = torch.empty(
            (num_embeddings, embedding_dim), device=device, dtype=dtype
        )
        self.weight = nn.Parameter(weight_tensor)
        self.device = device
        self.dtype = dtype

        torch.nn.init.trunc_normal_(
            self.weight, mean=0.0, std=1, a=-3 * 1, b=3 * 1
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        weight_tensor = torch.ones(self.d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x_float = x.to(dtype=torch.float32)

        variance = (x_float ** 2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)

        output = (x_float / rms)  * self.weight

        return output.to(in_dtype)

    
class PositionwiseFeedForward(nn.Module):

    def __init__(self, 
                 d_model: int,
                 d_ff: int | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        if self.d_ff is None:
            d_ff_calc = int((8 / 3) * self.d_model)
            self.d_ff = 64 * ((d_ff_calc + 63) // 64)

        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out = self.w1(x)
        gate = w1_out * torch.sigmoid(w1_out)

        value = self.w3(x)

        return self.w2(gate * value)


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, 
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None
    ):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        pos = torch.arange(max_seq_len, device=device)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        angles = torch.outer(pos, freqs)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        return torch.stack([-x2, x1], dim=-1).flatten(start_dim=-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        return (x * cos) + (self.rotate_half(x) * sin)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    M = x.max(dim=dim, keepdim=True).values
    x_shift = x - M
    exp_x_shift = torch.exp(x_shift)
    exp_sum = torch.sum(exp_x_shift, dim=dim, keepdim=True)

    return exp_x_shift / exp_sum

def scaled_dot_product_attention(Q: torch.Tensor, 
                                 K: torch.Tensor, 
                                 V: torch.Tensor, 
                                 mask: torch.Tensor | None = None) -> torch.Tensor:

    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)

    scaled_scores = scores / d_k ** 0.5

    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(~mask, float('-inf'))

    attn_weights = softmax(scaled_scores, dim=-1)

    output = attn_weights @ V

    return output

class CausalMultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int,
                 num_heads: int, 
                 rope: RotaryPositionalEmbedding | None = None,
                 device: torch.device | None = None
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.out_proj = Linear(d_model, d_model, device=device)
    
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        batch_size, seq_len, _ = x.shape

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        out = scaled_dot_product_attention(Q, K, V, mask=mask)

        out = out.transpose(1, 2).contiguous()

        out = out.view(batch_size, seq_len, self.d_model)

        output = self.out_proj(out)

        return output

class TransformerBlock(nn.Module):

    def __init__(self, d_model: int,
                 num_heads: int,
                 d_ff: int
    ):  

        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)

        x = x + self.ffn(self.ln2(x))

        return x

class TransformerLM(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
            for _ in range(num_layers)
        ])

        self.ln_f = RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_indices.shape

        x = self.token_embedding(in_indices)

        token_positions = torch.arange(seq_len, device=in_indices.device)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    max_logits = torch.max(logits, dim=-1, keepdim=True).values

    shifted_logits = logits - max_logits
    
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    loss_per_token = log_sum_exp - target_logits

    return loss_per_token.mean()

class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                p.data -= lr * weight_decay * p.data

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) 

                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-alpha_t)

        return loss

def learning_rate_schedule(
        t: int,
        alpha_max: float,
        alpha_min: float,
        T_w: int,
        T_c: int
) -> float:
    # 1. warm-up
    if t < T_w:
        return (t / T_w) * alpha_max

    # 3. post_annealing
    if t > T_c:
        return alpha_min

    # 2. cosine annealing
    progress = (t - T_w) / (T_c - T_w)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return alpha_min + cosine_decay * (alpha_max - alpha_min)

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> None:
    params_with_grad = [p for p in params if p.grad is not None]
    if not params_with_grad:
        return

    total_norm_sq = 0.0
    for p in params_with_grad:
        param_norm = p.grad.detach().norm(2)
        total_norm_sq += param_norm.item() ** 2

    total_norm = total_norm_sq ** 0.5

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.detach().mul_(scale)