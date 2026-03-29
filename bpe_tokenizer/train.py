import os
import regex as re

from cs336_basics.utils import (
    _dec,
    _pretokenize_chunks,
    find_chunk_boundaries,
    num_workers,
    save_vocab_and_merges,
)
from multiprocessing import Pool


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    num_merges = vocab_size - len(special_tokens) - 256

    special_pattern = "|".join(re.escape(token) for token in special_tokens)
    split_token = special_tokens[0].encode("utf-8")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, split_token)

    arguments = [
        (str(input_path), start, end, special_pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(num_workers) as pool:
        chunk_results = pool.map(_pretokenize_chunks, arguments)

    pre_tokenized = []
    for chunk in chunk_results:
        pre_tokenized.extend(chunk)

    sequences = [list(token.encode("utf-8")) for token in pre_tokenized]

    pair_counts = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            if p in pair_counts:
                pair_counts[p] += 1
            else:
                pair_counts[p] = 1

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda k: (pair_counts[k], vocab[k[0]], vocab[k[1]]))
        a, b = best_pair
        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        for seq in sequences:
            i = 0
            while i < len(seq) - 1:
                if seq[i] == a and seq[i + 1] == b:
                    left = seq[i - 1] if i > 0 else None
                    right = seq[i + 2] if i + 2 < len(seq) else None

                    if left is not None:
                        _dec(pair_counts, (left, a))
                    _dec(pair_counts, (a, b))
                    if right is not None:
                        _dec(pair_counts, (b, right))

                    seq[i] = new_id
                    del seq[i + 1]

                    if left is not None:
                        pair_counts[(left, new_id)] = pair_counts.get((left, new_id), 0) + 1
                    if right is not None:
                        pair_counts[(new_id, right)] = pair_counts.get((new_id, right), 0) + 1
                else:
                    i += 1

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    cwd = os.getcwd()
    vocab_path = os.path.join(cwd, "vocab.json")
    merges_path = os.path.join(cwd, "merges.txt")
    save_vocab_and_merges(vocab, merges, vocab_path, merges_path)

    return (vocab, merges)
