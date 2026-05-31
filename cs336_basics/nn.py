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
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

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
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        
        super().__init__()

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

"""
门控FFN
"""
def silu_fn(in_features: torch.Tensor) -> torch.Tensor:

    return in_features * torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.up_proj = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.gate_proj = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down_proj = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        up_gate_features = self.gate_proj(x)
        gate = silu_fn(up_gate_features)

        signal = self.up_proj(x)

        out_features = self.down_proj(gate * signal)

        return out_features


"""softmax"""

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:

    x_max = torch.max(x, dim=dim, keepdim=True).values

    x_stable = x - x_max

    x_exp = torch.exp(x_stable)

    exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)

    return x_exp / exp_sum

"""scaled_dot"""
def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None
) -> torch.Tensor:
    dk = Q.size(-1)

    scores = torch.einsum('...nk, ...mk -> ...nm', Q, K) / math.sqrt(dk)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    
    probs = softmax(scores, dim=-1)

    output = torch.einsum('...nm, ...mk -> ...nk', probs, V)

    return output

class RotaryPositionalEmbedding(nn.Module):
    
    def __init__(self, theta: float, dk: int, context_length: int, device=None):

        super().__init__()
        powers = torch.arange(0, dk, 2, device=device).float() / dk

        freqs = 1.0 / (theta ** powers)

        t = torch.arange(0, context_length, device=device)

        freqs_matrix = torch.outer(t, freqs)

        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output
    
"""
前向selfattention
"""

class CausalSelfAttention(nn.Module):
    
    def __init__(self, d_model: int, head_nums: int, context_length: int, 
                 theta: float, bias: int, 
                 device=None, dtype=None):
        assert d_model % head_nums == 0 , "d_model 必须能被 head_nums 整除"

        super().__init__()

        self.head_nums = head_nums
        self.d_model = d_model
        self.dk = d_model // head_nums

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.out_porj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None and context_length is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.dk, context_length, device=device)
        else:
            self.rope = None
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        q = rearrange(self.q_proj(x), '... s (h d) -> ... h s d', h=self.head_nums)
        k = rearrange(self.k_proj(x), '... s (h d) -> ... h s d', h=self.head_nums)
        v = rearrange(self.v_proj(x), '... s (h d) -> ... h s d', h=self.head_nums)

        s = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                
                batch_dims = x.shape[:-2]
                token_positions = torch.arange(s, device=x.device).expand(*batch_dims, s)
            
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))

        atten_out = scaled_dot_product_attention(q, k, v, mask=mask)

        atten_out = rearrange(atten_out, '... h s d -> ... s (h d)')
        output = self.out_porj(atten_out)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, context_length: int, head_nums: int, d_ff: int,
                 theta: float,
                 use_rsm: bool, norm_mode: str = "pre", ffn_mode: str = "swiglu", 
                 device=None, dtype=None):
        super().__init__()        
        self.use_rms = use_rsm
        self.norm_mode = norm_mode
        self.ffn_mode = ffn_mode

        self.atten = CausalSelfAttention(d_model=d_model, head_nums=head_nums, context_length=context_length, 
                                         theta=theta, device=device, dtype=dtype)

        if use_rsm:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        
        if self.ffn_mode == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif self.ffn_mode == "silu":
            d_ff = 4 * d_model
            self.ffn = nn.Sequential(
                Linear(d_model, d_ff, device, dtype),
                nn.SiLU(),
                Linear(d_ff, d_model, device, dtype)
            )
        else: 
            raise ValueError(f"Unknow ffn type: {ffn_mode}")
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        
        if self.norm_mode == "pre":
            x = x + self.atten(self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))
        elif self.norm_mode == "post":
            x = self.ln1(x + self.atten(x, token_positions=token_positions))
            x = self.ln2(x + self.ffn(x))
        
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 d_ff: int, rope_theta: int, head_nums: int, layer_nums: int,
                 use_rms: bool = True,
                 norm_mode: str = "pre", ffn_mode: str = "swiglu",
                 device=None, dtype=None):
        super().__init__()

        self.layers_num = layer_nums
        self.context_length = context_length

        self.emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, context_length, head_nums, d_ff, 
                             rope_theta, use_rms, norm_mode, ffn_mode,
                             device=device, dtype=dtype)
                             for _ in range(layer_nums)
        ])

        if use_rms:
            self.final_ln = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.final_ln = nn.Identity()

        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.tensor) -> torch.Tensor:
        b, s = token_ids.shape

        token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, s)

        x = self.emb(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        
        x = self.final_ln(x)

        return self.lm_head(x)
    
    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:

        sorted_logits, sorted_indecies = torch.sort(logits, dim=-1, descending=True)

        accum_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

        sorted_remove = accum_probs > p

        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False

        indecies_to_remove = torch.zeros_like(logits, torch.bool)
        indecies_to_remove = indecies_to_remove.scatter(1, sorted_indecies, sorted_remove)

        logits = logits.masked_fill(indecies_to_remove, float('-inf'))

        return logits

    @torch.no_grad()
    def generate(
        self, 
        prompt_ids: torch.Tensor,
        max_token_length: int,
        temperature: float,
        p: float,
        eos: int
    ):
        self.eval()

        generated = prompt_ids.clone()

        for _ in range(max_token_length):
            input_ = generated[..., -self.context_length:]

            logits = self.forward(input_)

            logits = logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / (temperature + 1e-8)
            
            if p < 1.0:
                logits = self._top_p_filter(logits, p)

            probs = softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, new_token), dim=1)

            if eos is not None and (new_token == eos).all():
                break
        
        return generated
