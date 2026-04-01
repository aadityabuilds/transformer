import math 
import torch 
import torch.nn as nn
from einops import rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype 
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=self.device, dtype=self.dtype))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return x @ self.weights.T 

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=torch.float32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=self.device, dtype=self.dtype))
        std = math.sqrt(2 / (num_embeddings + embedding_dim))
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids):
        return self.weights[token_ids]

class RoPE(nn.Module):
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

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model 
        self.gamma_weights = nn.Parameter(torch.ones(d_model))
        self.eps = eps 

    def forward(self, x):
        squared_mean = (torch.mean(x**2, dim=-1, keepdim=True))
        RMS = torch.sqrt(squared_mean + self.eps)
        return (x / RMS) * self.gamma_weights

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        pass

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        pass