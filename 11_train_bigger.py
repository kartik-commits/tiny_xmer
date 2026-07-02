"""
Part 7: A bigger, longer-trained model for better output quality.

Same architecture as before (reused from gpt_model.py) — just scaled up:
  n_embd    192 -> 384    (more capacity per token)
  block_size 128 -> 192   (sees more context)
  max_iters 3000 -> 5000  (trains longer)

Tuned to fit a 4GB GPU (GTX 1650): modest batch size, periodic checkpointing so
we keep the best model even if training is stopped early. Saves ckpt_big.pt.

NOTE: this is still CHARACTER-level, so it will still invent some non-words.
The big jump to real words comes from subword/BPE tokenization (a separate step).
"""
import torch
from torch.nn import functional as F
from gpt_model import GPT

# ---------------------------------------------------------------------------
# Config (scaled up, but sized for 4GB VRAM)
# ---------------------------------------------------------------------------
batch_size = 32
block_size = 192
max_iters = 5000
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
# Data
# ---------------------------------------------------------------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda nums: "".join(itos[i] for i in nums)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# Model (reused architecture)
# ---------------------------------------------------------------------------
model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
print(f"device: {device} | parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M",
      flush=True)

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
        "stoi": stoi, "itos": itos,
    }, "ckpt_big.pt")

# ---------------------------------------------------------------------------
# Train, keeping the best-val checkpoint
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
best_val = float("inf")
print("--- Training ---", flush=True)
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        tag = ""
        if losses["val"] < best_val:
            best_val = losses["val"]
            save_ckpt()
            tag = "  <- saved (best)"
        print(f"iter {it:4d} | train {losses['train']:.4f} | val {losses['val']:.4f}{tag}",
              flush=True)
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nBest val loss: {best_val:.4f} (saved to ckpt_big.pt)", flush=True)

# ---------------------------------------------------------------------------
# Sample from the best model (top-k for cleaner output)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(max_new_tokens=600, temperature=1.0, top_k=10):
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
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

print("\n--- Sample (top_k=10) ---", flush=True)
sample = generate()
print(sample, flush=True)
with open("sample_big.txt", "w") as f:
    f.write(sample)
