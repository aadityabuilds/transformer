import regex as re 
GPT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self):
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[int, int]] = []
        self.special_tokens: list[str] = []

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.special_tokens = special_tokens

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

        while len(self.vocab) < vocab_size - len(special_tokens): 
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
            
        for token in special_tokens: 
            self.vocab[len(self.vocab)] = token.encode("utf-8")

    def encode(self, text: str) -> list[int]:
        result = []
        for token in re.finditer(GPT_PATTERN, text):
            utf_encoding = list(token.group().encode("utf-8"))
            for i in range (len(self.merges)): 
                j = 0
                while j < len(utf_encoding) - 1:
                    if utf_encoding[j] == self.merges[i][0] and utf_encoding[j+1] == self.merges[i][1]:
                        utf_encoding[j] = 256 + i
                        del utf_encoding[j+1]
                    else:
                        j += 1
            result.extend(utf_encoding)
        return result

    def decode(self, token_ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in token_ids).decode("utf-8")