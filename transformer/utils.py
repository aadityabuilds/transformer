from collections.abc import Callable
from typing import Optional
import math 
import torch 
from torch.optim import Optimizer
from transformer import Transformer

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

def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w: 
        return t * alpha_max / T_w
    if t > T_c: 
        return alpha_min
    cos_term = math.cos((t - T_w) / (T_c - T_w) * math.pi)
    return alpha_min + 0.5 * (1 + cos_term) * (alpha_max - alpha_min)

def gradient_clipping(parameters, max_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]        
    norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
    if norm > max_norm:
        scaling_factor = max_norm / (norm + eps)
        for g in grads:
            g.detach().mul_(scaling_factor)

def data_loading(x, batch_size, context_length, device):
    starts = torch.randint(0, len(x) - context_length, (batch_size,))
    addition = torch.arange(context_length)
    starts = starts.unsqueeze(1)
    addition = addition.unsqueeze(0)
    x_tensor = torch.tensor(x)
    sequences = x_tensor[starts + addition]
    targets = x_tensor[starts + addition + 1]
    return sequences.to(device), targets.to(device)   

@torch.no_grad()
def evaluate(model, train_data, val_data, batch_size, context_length, device, eval_batches):
    model.eval()
    results = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        if data is None:
            continue
        total = 0.0
        for _ in range(eval_batches):
            x, y = data_loading(data, batch_size, context_length, device)
            logits = model(x)
            total += cross_entropy_loss(logits, y).item()
        results[name] = total / eval_batches
    model.train()
    return results

def save_checkpoint(model, optimizer, iteration, out):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    torch.save({
        "model_state": model_state,
        "optimizer_state": optimizer_state, 
        "iteration_state": iteration
    }, out)

def load_checkpoint(src, model, optimizer):
    content = torch.load(src)
    model.load_state_dict(content["model_state"])
    optimizer.load_state_dict(content["optimizer_state"])
    return content["iteration_state"]

def decoding(prompt, max_token, temperature=1.0, top_p=0.9):
    model = Transformer()
    output = ""
    END_OF_TEXT = "<|endoftext|>"
    for _ in range(max_token):
        token_list = tokenizer.encode(prompt)
        prediction = model.forward(token_list)
        next_token_logits = prediction[..., -1, :]
        shifted = next_token_logits / temperature
        probs = torch.softmax(shifted, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        indices_to_remove = (cumulative_probs - sorted_probs) > top_p
        sorted_probs[indices_to_remove] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        next_token_id = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices[next_token_id] 
        next_token = tokenizer.decode(next_token)
        if next_token == END_OF_TEXT:
            break
        prompt += next_token
        output += next_token
        
    return output