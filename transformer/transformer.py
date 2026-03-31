import math
from sympy.core import power
import torch
import torch.nn as nn
from einops import rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return x @ self.weights.T
 
class Embedding(nn.Module):
    def __init__(self, num_embedings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embedings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype 
        self.weights = nn.Parameter(torch.empty(num_embedings, embedding_dim, device=device, dtype=dtype))
        std = math.sqrt(2 / (num_embedings + embedding_dim))
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids):
        return self.weights[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype 
        self.weights = nn.Parameter(torch.empty(d_model))
        std = math.sqrt(2 / d_model)
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = x * result * self.weights
        return result.to(in_dtype)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.d_ff = d_ff
        self.weights1 = nn.Parameter(torch.empty(self.d_ff, d_model))
        self.weights3 = nn.Parameter(torch.empty(self.d_ff, d_model))
        self.weights2 = nn.Parameter(torch.empty(d_model, self.d_ff))
        std = math.sqrt(2 / d_model + self.d_ff)
        torch.nn.init.trunc_normal_(self.weights1, mean=0.0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.weights2, mean=0.0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.weights3, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        first_product = x @ self.weights1.T
        SiLU = first_product * torch.sigmoid(first_product)
        intermediate = SiLU * (x @ self.weights3.T)
        return intermediate @ self.weights2.T

class rope(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device 
        powers = torch.arange(0, d_k, 2) / d_k
        freqs = 1.0 / (theta ** powers)
        t = torch.arange(max_seq_len)
        angles = torch.outer(t, freqs)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
    
    def forward(self, x, token_positions):
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        if token_positions.ndim == 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        result = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        return result.flatten(-2)

def softmax(input, dim):
    maximum = torch.amax(input, dim, keepdim=True)
    input = torch.exp(input - maximum)
    denominator = torch.sum(input, dim=dim, keepdim=True)
    return input / denominator

def scaled_dot_product_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor, mask=None):
    if mask is not None: 
        d_k = Q.shape[-1]
        num = Q @ K.transpose(-2, -1)
        masked = num.masked_fill(~mask, float('-inf'))
        attention = masked / math.sqrt(d_k)
        return softmax(attention, dim=-1) @ V
    else:
        d_k = Q.shape[-1]
        masked = Q @ K.transpose(-2, -1)
        attention = masked / math.sqrt(d_k)
        return softmax(attention, dim=-1) @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads