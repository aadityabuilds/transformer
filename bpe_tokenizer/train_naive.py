import regex as re
import time  
GPT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):

    time1 = time.perf_counter()
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    SPECIAL_TOKEN_PATTERN = "|".join(re.escape(token) for token in special_tokens)
    processed = re.split(SPECIAL_TOKEN_PATTERN, text)

    pre_tokenized = []
    for chunk in processed:
        for match in re.finditer(GPT_PATTERN, chunk):
            pre_tokenized.append(match.group())
    
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    sequence = [list(token.encode("utf-8")) for token in pre_tokenized]

    while len(vocab) < vocab_size - len(special_tokens):
        pairs = {}
        for seq in sequence:
            for a, b in zip(seq, seq[1:]):
                if (a,b) in pairs:
                    pairs[(a,b)] += 1
                else:
                    pairs[(a,b)] = 1
        best_pair = max(pairs,key=lambda k: (pairs[k], vocab[k[0]], vocab[k[1]]))
        new_id = len(vocab)
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        for seq in sequence:
            i = 0
            while i < len(seq) - 1:
                if seq[i] == best_pair[0] and seq[i+1] == best_pair[1]:
                    seq[i] = new_id
                    del seq[i+1]
                i += 1

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    
    time2 = time.perf_counter()
    print(f"Finished in {time2 - time1}")
    return (vocab, merges)