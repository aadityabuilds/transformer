from collections.abc import Callable, Iterable
from typing import Optional
import torch 
import torch.nn as nn 
from torch import torch
from torch.optim import Optimizer

def cross_entropy_loss(logits, targets):
    max_logit = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logit
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    log_softmax = shifted - log_sum_exp
    target_log_probs = log_softmax.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            epsilon = group["eps"]
            lamda = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                state = self.state[p]

                if len(state) == 0: 
                    state["t"] = 1
                    state["v"] = torch.zeros_like(p.data, requires_grad=False)
                    state["m"] = torch.zeros_like(p.data, requires_grad=False)
                
                gradient = p.grad.data
                state["m"] = beta_1 * state["m"] + (1 - beta_1) * gradient 
                state["v"] = beta_2 * state["v"] + (1 - beta_2) * (gradient**2)
                adjusted_alpha = lr * ((1 - (beta_2**state["t"])) ** 0.5) / (1 - (beta_1**state["t"]))
                theta_old = p.data.clone()
                p.data = p.data - adjusted_alpha * state["m"] / (torch.sqrt(state["v"]) + epsilon)
                p.data = p.data - (lr * lamda * theta_old)
                state["t"] += 1
                
        return loss 