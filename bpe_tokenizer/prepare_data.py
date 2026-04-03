"""Convert a text file into a flat uint16 binary file using a trained BPE tokenizer.

Usage:
    uv run python bpe_tokenizer/prepare_data.py \
        --input data/TinyStoriesV2-GPT4-valid.txt \
        --tokenizer bpe_tokenizer/checkpoints/tinystories_valid.json \
        --output data/tinystories_valid.bin
"""

import argparse
import os
import sys
from time import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bpe_tokenizer.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Tokenize text into uint16 binary for transformer training")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to trained tokenizer JSON")
    parser.add_argument("--output", type=str, required=True, help="Output binary file path")
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="Characters to process at a time (for memory efficiency)")
    args = parser.parse_args()

    tokenizer = Tokenizer()
    tokenizer.load(args.tokenizer)
    vocab_size = len(tokenizer.vocab)
    print(f"Loaded tokenizer: {vocab_size} tokens")

    if vocab_size > 65535:
        raise ValueError(f"Vocab size {vocab_size} exceeds uint16 max (65535)")

    print(f"Reading {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  {len(text):,} characters")

    start = time()
    print("Encoding...")
    token_ids = tokenizer.encode(text)
    elapsed = time() - start
    print(f"  {len(token_ids):,} tokens in {elapsed:.1f}s")
    print(f"  compression ratio: {len(text) / len(token_ids):.2f} chars/token")

    arr = np.array(token_ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    arr.tofile(args.output)
    file_size = os.path.getsize(args.output)
    print(f"Saved to {args.output} ({file_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
