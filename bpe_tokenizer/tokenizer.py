import json
import regex as re
from cs336_basics.utils import gpt2_bytes_to_unicode, GPT_PATTERN

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        # Order-preserving unique list so regex alternation and lookups are unambiguous
        self.special_tokens = list(dict.fromkeys(special_tokens)) if special_tokens else []
        self._bytes_to_id = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        byte_decoder = {ch: b for b, ch in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as f:
            on_disk = json.load(f)
        vocab = {
            token_id: bytes(byte_decoder[ch] for ch in gpt2_str)
            for gpt2_str, token_id in on_disk.items()
        }

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip()
                if not cleaned:
                    continue
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    continue
                left, right = parts
                merges.append(
                    (
                        bytes(byte_decoder[ch] for ch in left),
                        bytes(byte_decoder[ch] for ch in right),
                    )
                )

        if special_tokens:
            for st in special_tokens:
                b = st.encode("utf-8")
                if b not in set(vocab.values()):
                    vocab[len(vocab)] = b

        return cls(vocab, merges, special_tokens)

    def encode(self, text):
        specials = self.special_tokens
        if specials:
            alt = "|".join(re.escape(t) for t in sorted(specials, key=len, reverse=True))
            parts = re.split(f"({alt})", text)
        else:
            parts = [text]

        output_token_ids = []
        special_set = set(specials)

        for part in parts:
            if part == "":
                continue
            if part in special_set:
                key = part.encode("utf-8")
                if key not in self._bytes_to_id:
                    raise ValueError(f"Special token {part!r} is not in the vocabulary")
                output_token_ids.append(self._bytes_to_id[key])
                continue

            for match in re.finditer(GPT_PATTERN, part):
                pretoken = match.group()
                seq = [bytes([b]) for b in pretoken.encode("utf-8")]
                for merge in self.merges:
                    i = 0
                    while i < len(seq) - 1:
                        if (seq[i], seq[i + 1]) == merge:
                            seq[i] = seq[i] + seq[i + 1]
                            del seq[i + 1]
                        else:
                            i += 1
                for token_bytes in seq:
                    if token_bytes not in self._bytes_to_id:
                        raise ValueError(f"Token bytes {token_bytes!r} not found in vocab")
                    output_token_ids.append(self._bytes_to_id[token_bytes])

        return output_token_ids

    def encode_iterable(self, iterable):
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids):
        return "".join(self.vocab[i].decode("utf-8", errors="replace") for i in ids)
