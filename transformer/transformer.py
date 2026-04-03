import math 
import torch 
import torch.nn as nn
from einops import rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return x @ self.weights.T 

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=torch.float32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
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

def SiLU(x):
    return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, device=None, dtype=torch.float32):
        super().__init__()
        self.d_model = d_model
        self.d_ff = int(d_model * 8 / 3)
        self.weight1 = Linear(in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.weight2 = Linear(in_features=self.d_ff, out_features=d_model, device=device, dtype=dtype)
        self.weight3 = Linear(in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype)

    def forward(self, x):
        first = self.weight1(x)
        first = SiLU(first)
        second = self.weight3(x)
        intermediate = first * second
        return self.weight2(intermediate)
        
def scaled_dot_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    numerator = Q @ K.transpose(-2, -1)
    term = numerator / math.sqrt(d_k)
    if mask is not None: 
        term = term.masked_fill(mask, float('-inf'))
    softmax = torch.softmax(term, dim=-1)
    return softmax @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None, use_rope=False):
        super().__init__()
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.d_k = d_model // num_heads

        self.W_Q = Linear(in_features=d_model, out_features=(self.d_k * num_heads))
        self.W_K = Linear(in_features=d_model, out_features=(self.d_k * num_heads))
        self.W_V = Linear(in_features=d_model, out_features=(self.d_k * num_heads))
        self.W_O = Linear(in_features=d_model, out_features=d_model)

        if use_rope:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device="cpu")
        
    def forward(self, x):
        seq_len = x.shape[-2]

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = rearrange(Q, "b n (num_heads d_k) -> b num_heads n d_k", num_heads = self.num_heads) 
        K = rearrange(K, "b n (num_heads d_k) -> b num_heads n d_k", num_heads = self.num_heads) 
        V = rearrange(V, "b n (num_heads d_k) -> b num_heads n d_k", num_heads = self.num_heads) 

        if self.use_rope:
            token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attention = scaled_dot_attention(Q, K, V, mask)

        attention = rearrange(attention, "b num_heads n d_k ->  b n (num_heads d_k)", num_heads=self.num_heads)
        result = self.W_O(attention)
        return result 

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None, use_rope=False):
        super().__init__()
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.MHA = MultiHeadAttention(self.d_model, self.num_heads, self.theta, self.max_seq_len, self.use_rope)
        self.FFN = FeedForward(self.d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        first_norm = self.norm1(x)
        MHA = self.MHA(first_norm)
        x = x + MHA 
        second_norm = self.norm2(x)
        FFN = self.FFN(second_norm)
        return x + FFN 

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads,  num_embeddings, embedding_dim, theta=None, max_seq_len=None, use_rope=False):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(num_embeddings, embedding_dim)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, theta, max_seq_len, use_rope) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.linear = Linear(in_features=d_model, out_features=num_embeddings)

    def forward(self, token_list):
        embeddings = self.embedding(token_list)
        for block in self.blocks: 
            embeddings = block(embeddings)
        normalized = self.norm(embeddings)
        logits = self.linear(normalized)
        return logits 