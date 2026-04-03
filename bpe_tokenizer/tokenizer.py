from __future__ import annotations

import json
import os
import multiprocessing as mp
from time import time

import regex as re
from tqdm import tqdm

from bpe_tokenizer.utils import find_chunk_boundaries, MergeStats, TrainingTracker

GPT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pretokenize_chunk(input_path: str, special_pattern: str, start: int, end: int) -> list[str]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    text = chunk_bytes.decode("utf-8", errors="replace")
    parts = re.split(special_pattern, text) if special_pattern else [text]
    result = []
    for part in parts:
        for m in re.finditer(GPT_PATTERN, part):
            result.append(m.group())
    return result


class Tokenizer:
    def __init__(self):
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[int, int]] = []
        self.special_tokens: list[str] = []

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str],
              tracker: TrainingTracker | None = None):
        self.special_tokens = special_tokens
        if tracker:
            tracker.on_train_start()

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        processed = re.split(special_pattern, text)
        pre_tokenized = []

        for chunk in processed:
            tokenized = re.finditer(GPT_PATTERN, chunk)
            for token in tokenized:
                pre_tokenized.append(token.group())

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        sequences = []
        for chunk in pre_tokenized:
            sequences.append(list(chunk.encode("utf-8")))

        num_merges = vocab_size - len(special_tokens) - 256
        pbar = tqdm(total=num_merges, desc="Training (simple)", unit="merge")

        while len(self.vocab) < vocab_size - len(special_tokens):
            if tracker:
                tracker.on_merge_start()

            pairs = {}
            for token_bytes in sequences:
                for a, b in zip(token_bytes, token_bytes[1:]):
                    if (a, b) in pairs:
                        pairs[(a, b)] += 1
                    else:
                        pairs[(a, b)] = 1

            best_pair = max(pairs, key=lambda k: (pairs[k], self.vocab[k[0]], self.vocab[k[1]]))
            new_id = len(self.vocab)

            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges.append((best_pair[0], best_pair[1]))

            for token_bytes in sequences:
                i = 0
                while i < len(token_bytes) - 1:
                    if token_bytes[i] == best_pair[0] and token_bytes[i + 1] == best_pair[1]:
                        token_bytes[i] = new_id
                        del token_bytes[i + 1]
                    else:
                        i += 1

            merge_time = time() - tracker.merge_start_time if tracker else 0.0
            merge_idx = len(self.merges) - 1

            if tracker:
                stats = MergeStats(
                    merge_index=merge_idx,
                    pair=best_pair,
                    new_token=self.vocab[new_id],
                    pair_count=pairs[best_pair],
                    merge_time=merge_time,
                    total_elapsed=tracker.elapsed(),
                    vocab_size=len(self.vocab),
                    num_active_pairs=len(pairs),
                )
                tracker.on_merge_end(stats)

            pbar.set_postfix(pair=f"{self.vocab[best_pair[0]]!r}+{self.vocab[best_pair[1]]!r}",
                             freq=pairs[best_pair])
            pbar.update(1)

        pbar.close()

        for token in special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")

        if tracker:
            tracker.on_train_end(len(self.merges))

    def train_efficient(self, input_path: str, vocab_size: int, special_tokens: list[str],
                        num_workers: int | None = None, tracker: TrainingTracker | None = None):
        self.special_tokens = special_tokens
        if tracker:
            tracker.on_train_start()

        if num_workers is None:
            num_workers = os.cpu_count() or 4

        special_pattern = "|".join(re.escape(t) for t in special_tokens) if special_tokens else ""
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_workers, split_token)

        chunk_args = [(input_path, special_pattern, boundaries[i], boundaries[i + 1])
                      for i in range(len(boundaries) - 1)
                      if boundaries[i] < boundaries[i + 1]]

        with mp.Pool(num_workers) as pool:
            chunk_results = pool.starmap(_pretokenize_chunk, chunk_args)

        pre_tokenized = []
        for result in chunk_results:
            pre_tokenized.extend(result)

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        sequences = [list(chunk.encode("utf-8")) for chunk in pre_tokenized]

        pairs: dict[tuple[int, int], int] = {}
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + 1

        num_merges = vocab_size - len(special_tokens) - 256
        pbar = tqdm(total=num_merges, desc="Training (efficient)", unit="merge")

        while len(self.vocab) < vocab_size - len(special_tokens):
            if tracker:
                tracker.on_merge_start()

            best_pair = max(
                (p for p in pairs if pairs[p] > 0),
                key=lambda k: (pairs[k], self.vocab[k[0]], self.vocab[k[1]]),
            )
            A, B = best_pair
            best_count = pairs[best_pair]
            new_id = len(self.vocab)

            self.vocab[new_id] = self.vocab[A] + self.vocab[B]
            self.merges.append((A, B))

            for seq in sequences:
                i = 0
                while i < len(seq) - 1:
                    if seq[i] == A and seq[i + 1] == B:
                        if i > 0:
                            pairs[(seq[i - 1], A)] -= 1
                        if i + 2 < len(seq):
                            pairs[(B, seq[i + 2])] -= 1
                        pairs[(A, B)] -= 1

                        seq[i] = new_id
                        del seq[i + 1]

                        if i > 0:
                            pairs[(seq[i - 1], new_id)] = pairs.get((seq[i - 1], new_id), 0) + 1
                        if i + 1 < len(seq):
                            pairs[(new_id, seq[i + 1])] = pairs.get((new_id, seq[i + 1]), 0) + 1
                    else:
                        i += 1

            merge_time = time() - tracker.merge_start_time if tracker else 0.0
            merge_idx = len(self.merges) - 1
            active_pairs = sum(1 for v in pairs.values() if v > 0)

            if tracker:
                stats = MergeStats(
                    merge_index=merge_idx,
                    pair=best_pair,
                    new_token=self.vocab[new_id],
                    pair_count=best_count,
                    merge_time=merge_time,
                    total_elapsed=tracker.elapsed(),
                    vocab_size=len(self.vocab),
                    num_active_pairs=active_pairs,
                )
                tracker.on_merge_end(stats)

            pbar.set_postfix(pair=f"{self.vocab[A]!r}+{self.vocab[B]!r}", freq=best_count)
            pbar.update(1)

        pbar.close()

        for token in special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")

        if tracker:
            tracker.on_train_end(len(self.merges))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        self.merges = [tuple(p) for p in data["merges"]]
        self.special_tokens = data["special_tokens"]

    def encode(self, text: str) -> list[int]:
        result = []
        for token in re.finditer(GPT_PATTERN, text):
            utf_encoding = list(token.group().encode("utf-8"))
            for i in range(len(self.merges)):
                j = 0
                while j < len(utf_encoding) - 1:
                    if utf_encoding[j] == self.merges[i][0] and utf_encoding[j + 1] == self.merges[i][1]:
                        utf_encoding[j] = 256 + i
                        del utf_encoding[j + 1]
                    else:
                        j += 1
            result.extend(utf_encoding)
        return result

    def decode(self, token_ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in token_ids).decode("utf-8")
