import torch 
import torch.nn as nn 
from torch.profiler import profile, ProfilerActivity, record_function 
from transformer.transformer import Transformer 

def main(): 

    model = Transformer(num_layers=4, d_model=1024, num_heads=4,
        vocab_size=10000, theta=10000, max_seq_len=512, use_rope=True
    ).cuda()

    token_list = torch.randint(0, 10000, (4,128)).cuda()

    for _ in range(2):
        out = model(token_list)
        out.sum().backward()
        model.zero_grad()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        for _ in range(10):
            with record_function("forward"):
                out = model(token_list)
                loss = out.sum()
            torch.cuda.synchronize()

            with record_function("backward"):
                loss.backward()
            torch.cuda.synchronize()
            model.zero_grad()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))

if __name__ == "__main__":
    main()