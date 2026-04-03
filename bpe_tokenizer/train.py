import argparse
import sys
import os
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bpe_tokenizer.tokenizer import Tokenizer
from bpe_tokenizer.utils import TrainingTracker


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Path to training text file")
    parser.add_argument("--vocab-size", type=int, default=512, help="Target vocabulary size")
    parser.add_argument("--special-tokens", type=str, nargs="*", default=["<|endoftext|>"],
                        help="Special tokens to reserve")
    parser.add_argument("--mode", type=str, choices=["simple", "efficient"], required=True,
                        help="Training mode: simple or efficient")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of workers for efficient mode (default: cpu count)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save trained tokenizer (JSON)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="bpe-tokenizer",
                        help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Wandb run name (auto-generated if not set)")
    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        import wandb
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        run_name = args.wandb_run_name or f"{args.mode}_{input_name}_v{args.vocab_size}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "mode": args.mode,
                "input": args.input,
                "vocab_size": args.vocab_size,
                "special_tokens": args.special_tokens,
                "num_workers": args.num_workers,
            },
        )

    tracker = TrainingTracker(wandb_run=wandb_run)
    tokenizer = Tokenizer()

    print(f"Mode: {args.mode}")
    print(f"Input: {args.input}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Special tokens: {args.special_tokens}")
    if args.wandb:
        print(f"Wandb: {wandb_run.url}")

    start = time()

    if args.mode == "simple":
        tokenizer.train(args.input, args.vocab_size, args.special_tokens, tracker=tracker)
    else:
        tokenizer.train_efficient(args.input, args.vocab_size, args.special_tokens,
                                  num_workers=args.num_workers, tracker=tracker)

    elapsed = time() - start
    print(f"\nTraining complete.")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Merges: {len(tokenizer.merges)}")
    print(f"Time elapsed: {elapsed:.2f}s")

    if args.output:
        tokenizer.save(args.output)
        print(f"Saved tokenizer to {args.output}")

    test_text = "Once upon a time, there was a little girl."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nSanity check:")
    print(f"  Input:   {test_text!r}")
    print(f"  Encoded: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
    print(f"  Decoded: {decoded!r}")
    print(f"  Match:   {test_text == decoded}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
