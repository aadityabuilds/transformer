import argparse
import sys
import os
import torch
from transformer import Transformer
from transformer.utils import generate
from bpe_tokenizer.tokenizer import Tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to BPE tokenizer JSON")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, mps, or cuda")

    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Vocab size (auto-detected from tokenizer if omitted)")
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--use_rope", action="store_true", default=True)

    args = parser.parse_args()
    device = get_device(args.device)
    print(f"device: {device}")

    tokenizer = Tokenizer()
    tokenizer.load(args.tokenizer)
    vocab_size = args.vocab_size or len(tokenizer.vocab)
    print(f"tokenizer: {len(tokenizer.vocab)} tokens, using vocab_size={vocab_size}")

    model = Transformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        vocab_size=vocab_size,
        theta=args.theta,
        max_seq_len=args.context_length,
        use_rope=args.use_rope,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters (step {ckpt['iteration_state']})")

    print(f"\nprompt: {args.prompt!r}")
    print(f"temperature: {args.temperature}, top_p: {args.top_p}, max_tokens: {args.max_tokens}")
    print("-" * 60)

    output = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    print(output)

if __name__ == "__main__":
    main()