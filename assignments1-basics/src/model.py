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