"""
Part 8 (Step 1): Byte-Pair Encoding (BPE) tokenizer — from scratch.

The char-level model spelled words letter-by-letter and invented fake spellings.
BPE (what GPT-2/3/4 use) instead learns a vocabulary of frequent CHUNKS:

  1. Start from raw UTF-8 bytes  -> 256 base tokens.
  2. Count all adjacent token pairs; merge the MOST FREQUENT pair into a new
     token id (256, 257, ...).
  3. Repeat. Common sequences ("th", "the", " and", "ing") become single tokens.

The model then predicts word-pieces, so its output is real fragments by design.
This script trains the tokenizer on input.txt and saves it to bpe_tokenizer.pkl.
"""
import pickle

# ---------------------------------------------------------------------------
# The two core operations
# ---------------------------------------------------------------------------
def get_stats(ids):
    """Count how often each adjacent pair (a, b) occurs."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in ids with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

# ---------------------------------------------------------------------------
# Train BPE: keep merging the most frequent pair until we hit the target vocab
# ---------------------------------------------------------------------------
VOCAB_SIZE = 512                     # 256 base bytes + 256 learned merges
num_merges = VOCAB_SIZE - 256

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

ids = list(text.encode("utf-8"))     # raw bytes, each 0..255
print(f"Corpus: {len(text)} chars -> {len(ids)} byte-tokens to start")

merges = {}                          # (a, b) -> new_id, in the order learned
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)  # most frequent adjacent pair
    new_id = 256 + i
    ids = merge(ids, pair, new_id)
    merges[pair] = new_id
    if (i + 1) % 64 == 0:
        print(f"  merge {i + 1:3d}/{num_merges}: {pair} -> {new_id}", flush=True)

# Build the decoding table: each token id -> its raw bytes
vocab = {idx: bytes([idx]) for idx in range(256)}
for (a, b), new_id in merges.items():
    vocab[new_id] = vocab[a] + vocab[b]

# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------
def encode(s):
    ids = list(s.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        # merge the pair with the LOWEST merge index that still exists (i.e.
        # apply merges in the same order they were learned)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Show what it learned
# ---------------------------------------------------------------------------
print(f"\nFinal vocab size: {len(vocab)}")
print("A few learned multi-byte tokens (id -> text):")
for idx in list(range(256, 256 + 12)) + [300, 400, 500]:
    if idx in vocab:
        print(f"  {idx}: {vocab[idx].decode('utf-8', errors='replace')!r}")

sample = "First Citizen: Before we proceed, hear me speak."
enc = encode(sample)
print(f"\nSample: {sample!r}")
print(f"  chars : {len(sample)}")
print(f"  tokens: {len(enc)}  -> {enc}")
print(f"  pieces: {[vocab[i].decode('utf-8', errors='replace') for i in enc]}")
print(f"  round-trip ok: {decode(enc) == sample}")

# Compression over the whole corpus
full = encode(text)
print(f"\nWhole corpus: {len(text)} chars -> {len(full)} tokens "
      f"({len(text)/len(full):.2f} chars/token)")

# Save the trained tokenizer for the model script
with open("bpe_tokenizer.pkl", "wb") as f:
    pickle.dump({"merges": merges, "vocab": vocab}, f)
print("\nSaved tokenizer to bpe_tokenizer.pkl")
