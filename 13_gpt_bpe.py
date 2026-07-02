"""
Part 8 (Step 2): Train the GPT on BPE (subword) tokens.

Same architecture as before (from gpt_model.py) — the ONLY change is the
tokenizer. We load the BPE tokenizer trained in 12_bpe.py, encode the corpus into
subword tokens, and train. Output should now be made of REAL words.

Losses are NOT comparable to the char-level model (per-token, vocab 512 vs 65);
we also report bits-per-character for a fair comparison. Saves ckpt_bpe.pt.
"""
import pickle
import math
import torch
from torch.nn import functional as F
from gpt_model import GPT

# ---------------------------------------------------------------------------
# Load the BPE tokenizer and rebuild encode/decode
# ---------------------------------------------------------------------------
with open("bpe_tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)
merges, vocab = tok["merges"], tok["vocab"]
vocab_size = len(vocab)

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

def encode(s):
    ids = list(s.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
batch_size = 32
block_size = 128        # 128 subword tokens ~= 250 chars of context
max_iters = 3500
eval_interval = 250
eval_iters = 100
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# ---------------------------------------------------------------------------
# Data: encode the whole corpus with BPE
# ---------------------------------------------------------------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
print("Encoding corpus with BPE...", flush=True)
data = torch.tensor(encode(text), dtype=torch.long)
chars_per_token = len(text) / len(data)
print(f"{len(text)} chars -> {len(data)} tokens ({chars_per_token:.2f} chars/token)",
      flush=True)

n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
print(f"device: {device} | vocab: {vocab_size} | "
      f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M", flush=True)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def save_ckpt():
    torch.save({
        "model_state": model.state_dict(),
        "config": dict(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head,
                       n_layer=n_layer, block_size=block_size, dropout=dropout),
    }, "ckpt_bpe.pt")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
best_val = float("inf")
print("--- Training ---", flush=True)
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        # bits-per-char = fair cross-tokenizer metric
        bpc = (losses["val"] / math.log(2)) / chars_per_token
        tag = ""
        if losses["val"] < best_val:
            best_val = losses["val"]; save_ckpt(); tag = "  <- saved (best)"
        print(f"iter {it:4d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"| val bits/char {bpc:.3f}{tag}", flush=True)
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nBest val loss: {best_val:.4f} "
      f"(bits/char {(best_val/math.log(2))/chars_per_token:.3f}) -> ckpt_bpe.pt", flush=True)

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(max_new_tokens=300, temperature=1.0, top_k=20):
    idx = torch.tensor([encode("\n")], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
    return decode(idx[0].tolist())

print("\n--- Sample (BPE model, top_k=20) ---", flush=True)
sample = generate()
print(sample, flush=True)
with open("sample_bpe.txt", "w") as f:
    f.write(sample)
