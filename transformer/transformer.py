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
    def __init__(self): 
        super().__init__()
        pass

class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
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