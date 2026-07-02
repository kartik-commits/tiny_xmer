"""
Part 5: The full GPT — assemble everything and train it.

New vs Part 4:
  - Token embedding + POSITION embedding (attention is position-blind on its own)
  - A stack of Blocks, a final LayerNorm, and a linear head to vocab logits
  - Dropout for regularization
  - generate() crops context to the last block_size tokens

Architecture:
  idx -> tok_emb + pos_emb -> [Block] x n_layer -> LayerNorm -> Linear -> logits
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Hyperparameters (sized to train in ~1-3 min on a GPU, yet clearly beat bigram)
# ---------------------------------------------------------------------------
batch_size = 64
block_size = 128       # context length (was 8 in the toy examples)
max_iters = 3000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
n_embd = 192           # embedding dimension C
n_head = 6             # attention heads (each 192/6 = 32 dim)
n_layer = 6            # number of transformer blocks stacked
dropout = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# ---------------------------------------------------------------------------
# Data (Parts 1-2)
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
# Model components (Parts 3-4, now with dropout)
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# ---------------------------------------------------------------------------
# The GPT
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)   # what a token IS
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # WHERE it sits
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)                # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)    # -> vocab logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                      # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb          # add identity + position -> (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)       # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]      # crop to last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]            # last time step -> (B, C)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

model = GPT().to(device)
print(f"device: {device}")
print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

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

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("\n--- Training ---")
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {it:4d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------------------------------------------------------------------
# Generate a sample and save it
# ---------------------------------------------------------------------------
print("\n--- Generated Shakespeare (500 chars) ---")
start = torch.zeros((1, 1), dtype=torch.long, device=device)
sample = decode(model.generate(start, max_new_tokens=500)[0].tolist())
print(sample)

with open("sample_output.txt", "w") as f:
    f.write(decode(model.generate(start, max_new_tokens=2000)[0].tolist()))
print("\n(Wrote a longer 2000-char sample to sample_output.txt)")

# Save a checkpoint so later scripts (Part 6) can load the trained model
# instead of retraining. We store the weights and the config needed to rebuild.
torch.save({
    "model_state": model.state_dict(),
    "config": dict(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head,
                   n_layer=n_layer, block_size=block_size, dropout=dropout),
    "stoi": stoi, "itos": itos,
}, "ckpt.pt")
print("(Saved checkpoint to ckpt.pt)")
