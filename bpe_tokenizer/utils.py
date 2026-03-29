import json
import os
import regex as re
from functools import lru_cache
from multiprocessing import Pool

GPT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
num_workers = os.cpu_count() or 1


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Maps each byte 0..255 to a printable Unicode string (GPT-2 style)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(x) for x in cs]
    return dict(zip(bs, characters))


def bytes_to_gpt2_str(b: bytes) -> str:
    enc = gpt2_bytes_to_unicode()
    return "".join(enc[x] for x in b)


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str,
    merges_path: str,
) -> None:
    vocab_json = {bytes_to_gpt2_str(token_bytes): idx for idx, token_bytes in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{bytes_to_gpt2_str(left)} {bytes_to_gpt2_str(right)}\n")


def find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pretokenize_chunks(args):
    path, start, end, special_pattern = args
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    text = raw.decode("utf-8")
    processed = re.split(special_pattern, text)

    pre_tokenized = []
    for chunk in processed:
        for match in re.finditer(GPT_PATTERN, chunk):
            pre_tokenized.append(match.group())

    return pre_tokenized


def _dec(pair_counts, pair):
    pair_counts[pair] -= 1
    if pair_counts[pair] <= 0:
        del pair_counts[pair]
