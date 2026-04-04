# BPE Tokenization from Scratch

## Why do we need tokenization?

Language models don't see text. They see numbers. Before a transformer can learn anything about language, we need a way to convert raw text into a sequence of integers and back again. This mapping is what a tokenizer does.

The question is: what should each integer represent?

## The obvious approaches, and where they fail

The simplest idea is **character-level encoding**. Map each character to an integer: `a=0, b=1, c=2, ...` and feed individual characters to the model. This works, but it has a serious problem. The word `" the"` becomes 4 tokens. The sentence `"Once upon a time, there was a little girl."` becomes 43 tokens. The model has to learn that `t`, `h`, `e` next to each other means "the" from raw statistics alone. Sequences become very long, and the model has to waste capacity learning basic spelling.

The opposite extreme is **word-level encoding**. Give each word its own integer. `" the"=0, " cat"=1, " running"=2, ...` Now sequences are short, but the vocabulary explodes. English has hundreds of thousands of words. Misspellings, rare names, code, other languages, every new word you encounter needs a new entry. You also lose the relationship between `"run"`, `"running"`, and `"runner"` since they're completely different integers.

What we want is something in between. Common words should be single tokens. Rare words should be broken into meaningful pieces. And the vocabulary should be a fixed, manageable size.

## The BPE algorithm

The [BPE algorithm](https://en.wikipedia.org/wiki/Byte-pair_encoding) was introduced in 1994 for text compression. It works with a simple idea: repeatedly find the most frequent pair of adjacent symbols and merge them into a new symbol.

Imagine you have the sequence `aaabdaaabac`. The pair `aa` occurs the most, so we replace it with a new symbol `Z=aa` to get `ZabdZabac`. Then `Za` is most frequent, so `Y=Za` gives us `YbdYbac`. We keep going until we reach a desired vocabulary size.

### Starting with bytes

Before we can run BPE, we need a base vocabulary. We use [UTF-8](https://en.wikipedia.org/wiki/UTF-8) encoding to convert text to bytes. UTF-8 is a variable-length encoding within the [Unicode](https://en.wikipedia.org/wiki/Unicode) standard. ASCII characters (basic Latin letters, digits, punctuation) fit in a single byte. Other languages and emoji use 2-4 bytes.

```python
text = "hello"
print(list(text.encode("utf-8")))
# [104, 101, 108, 108, 111]
```

This gives us a base vocabulary of 256 possible byte values. Every possible text can be represented as a sequence of these 256 tokens. BPE then builds on top of this base by merging frequent pairs into new tokens.

### Pre-tokenization

Before running BPE, we split the text into chunks using a regex pattern. This prevents merges from crossing word boundaries. We use the same GPT-style pattern:

```python
GPT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

This splits text into words (with leading spaces attached), contractions, numbers, and punctuation. The sentence `"I don't like cats"` becomes `["I", " don", "'t", " like", " cats"]`. Each chunk is then encoded to bytes independently, and BPE merges only happen within chunks, never across them.

We also split on special tokens like `<|endoftext|>` which act as document boundaries in the training data. These get reserved IDs at the end of the vocabulary and are never merged.

## Training the tokenizer

### The naive approach

The straightforward implementation of BPE training has two steps per merge iteration:

1. **Count all adjacent pairs** across every sequence
2. **Apply the merge** everywhere the winning pair appears

```python
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
```

The problem is that **pair counting is rebuilt from scratch every iteration**. With `T` total tokens and `M` merges, the counting step alone is `O(M x T)` dict operations. The counting pass also happens to be the more expensive of the two per-element since it involves a hash + lookup + increment for every adjacent pair in every sequence.

### The efficient approach

The key insight is that most pairs don't change between iterations. When we merge `(A, B)` into `AB`, only the pairs adjacent to merge sites are affected. So we build the pair frequency table once and update it incrementally.

When merging `(A, B)` at some position in a sequence `..., x, A, B, y, ...`:

- **Decrement**: `(x, A)`, `(A, B)`, and `(B, y)` each lose one count
- **Increment**: `(x, AB)` and `(AB, y)` each gain one count

That's 5 dict updates per merge site instead of rescanning millions of tokens.

```python
pairs: dict[tuple[int, int], int] = {}
for seq in sequences:
    for a, b in zip(seq, seq[1:]):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1

while len(self.vocab) < vocab_size - len(special_tokens):
    best_pair = max(
        (p for p in pairs if pairs[p] > 0),
        key=lambda k: (pairs[k], self.vocab[k[0]], self.vocab[k[1]]),
    )
    A, B = best_pair
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
```

On the TinyStories validation set (22MB), training a 10K vocabulary:

| Method | Time |
|--------|------|
| Naive | 373s (6.2 min) |
| Efficient | 103s (1.7 min) |
| **Speedup** | **3.6x** |

The efficient version also uses multiprocessing for the pre-tokenization step. The file is split into chunks aligned on special token boundaries using `find_chunk_boundaries`, and each worker applies the regex pattern independently. The results are concatenated before the merge loop begins.

## Encoding

Encoding converts raw text into token IDs using the learned merges. The naive approach iterates through all merges in order and scans the sequence for each one. With a 10K vocabulary that's 9,743 iterations per regex-matched word, most of which find nothing.

The efficient encoder flips the approach. Instead of asking "does merge #0 apply? does merge #1 apply? ..." for all 9,743 merges, it asks: "which pair currently in this sequence has the highest priority?"

```python
def _encode_chunk(text, merge_lookup):
    result = []
    for m in re.finditer(GPT_PATTERN, text):
        ids = list(m.group().encode("utf-8"))
        while len(ids) >= 2:
            best_idx = float("inf")
            best_pos = -1
            for j in range(len(ids) - 1):
                pair = (ids[j], ids[j + 1])
                idx = merge_lookup.get(pair)
                if idx is not None and idx < best_idx:
                    best_idx = idx
                    best_pos = j
            if best_pos == -1:
                break
            ids[best_pos] = 256 + best_idx
            del ids[best_pos + 1]
        result.extend(ids)
    return result
```

The `merge_lookup` is a dict mapping `(a, b) -> merge_index` built once from the merge list. For each word, we scan only the pairs that actually exist and pick the one with the lowest index (highest priority). A typical word of 8 bytes does ~4 rounds of this instead of 9,743 merge scans.

For encoding full files, `encode_file` adds multiprocessing on top. The file is split into byte-range chunks, and each worker encodes its chunk independently.

Encoding the 22MB validation set:

| Method | Time |
|--------|------|
| Naive | Would take 10+ minutes |
| Efficient + multiprocessing | **1.5 seconds** |

The 2.2GB training set encoded in 3 minutes.

## Decoding

Decoding is the reverse: convert token IDs back to text. Unlike encoding, it's already efficient.

```python
def decode(self, token_ids):
    return b"".join(self.vocab[i] for i in token_ids).decode("utf-8")
```

Each token ID maps directly to its byte sequence in the vocabulary table. Token 426 maps to `b"Once"`, token 436 maps to `b" upon"`. One O(1) dict lookup per token, concatenate the bytes, decode to UTF-8. There's no merge logic involved because the vocabulary table already stores the final byte sequence for every token. All the merge history from training is baked into those entries.

## Training results

We trained a 10K vocabulary BPE tokenizer on the TinyStories validation set (22MB, 27,630 stories separated by `<|endoftext|>` tokens). Training took 18 minutes with the efficient implementation.

The first merges learned are exactly what you'd expect from English text:

| Rank | Merge | Result |
|------|-------|--------|
| 1 | `' '` + `'t'` | `' t'` |
| 2 | `'h'` + `'e'` | `'he'` |
| 3 | `' '` + `'a'` | `' a'` |
| 6 | `' t'` + `'he'` | `' the'` |
| 10 | `' t'` + `'o'` | `' to'` |
| 11 | `' a'` + `'nd'` | `' and'` |

The pattern is textbook BPE. First it learns space-letter pairs (since English words are space-delimited). Then common bigrams like `he`, `nd`, `ed`. Then it composes those into full words like ` the`, ` and`, ` to`. The leading space is attached to the word, the same convention GPT tokenizers use.

After training, the tokenizer achieves a compression ratio of 4.1 bytes per token on TinyStories. The sentence `"Once upon a time, there was a little girl."` (43 characters) encodes to just 11 tokens: `[426, 436, 258, 395, 44, 400, 282, 258, 388, 472, 46]`.

The trained vocabulary and merge list are saved as JSON and can be loaded to encode new text or decode token sequences without retraining.
