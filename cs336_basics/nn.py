import torch
import torch.nn as nn
import math
from einops import rearrange

"""
创建全连接层
"""

class Linear(nn.Module):
    # 初始化
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        # 封装一下输出的device和dtype
        factory_kwargs = {"device": device, "dtype":dtype}

        # 创建权重weight
        self.weight = nn.parameter(torch.empty((in_features, out_features), **factory_kwargs))

        # 初始化权重
        # xavier
        std = (2.0 / (in_features + out_features)) ** 0.5

        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    
    # 前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...i, oi -> ...o', x, self.weight)
    
"""
创建embedding层
"""
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dims: int, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dims), **factory_kwargs))

        std = 1.0

        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a= -3 * std, b= 3 * std)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        
        return self.weight[token_ids]

"""
归一化层
"""
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, embedding_dims: int, eps: float = 1e-5, device=None, dtype=None):
        
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

        self.eps = eps
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x_float = x.to(torch.float32)

        ms = x_float.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        
        result = (x_float / rms) * self.weight

        return result.to(in_dtype)