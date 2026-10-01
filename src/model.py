import torch
import torch.nn as nn
from einops import einsum

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


class softmax(nn.Module):

    def __init__(self, dim: int | None = None
    ):
        super().__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor):
        M = x.max(dim=self.dim, keepdim=True).values
        x_shift = x - M
        exp_x_shift = torch.exp(x_shift)
        exp_sum = torch.sum(exp_x_shift, dim=self.dim, keepdim=True)

        return exp_x_shift / exp_sum