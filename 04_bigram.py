"""
Part 2 (Step 2): The Bigram model + a full training loop.

The bigram model predicts the next character using ONLY the current character
(no attention, no memory of earlier context). It's deliberately dumb — the point
is to see the COMPLETE training skeleton we'll reuse for the real transformer:
    forward pass -> loss -> backward pass (gradients) -> optimizer step.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# Setup: tokenizer + data + batching (from Parts 1 and 2, Step 1)
# ----------------------------------------------------------------------------
torch.manual_seed(1337)

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

# Hyperparameters
batch_size = 32
block_size = 8
learning_rate = 1e-2
max_steps = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # An embedding table of shape (vocab_size, vocab_size).
        # Row i = the scores ("logits") for what token comes after token i.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx is (B, T) integers. Looking them up gives (B, T, vocab_size):
        # for every position, a score for each possible next character.
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # cross_entropy wants (N, C) logits and (N,) targets, so flatten
            # the batch and time dims together.
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # Cross-entropy loss: how surprised is the model by the true next
            # char? Lower = better. Random guessing over 65 classes ~= 4.17.
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T); we extend it max_new_tokens times.
        for _ in range(max_new_tokens):
            logits, _ = self(idx)            # (B, T, C)
            logits = logits[:, -1, :]        # keep only the LAST time step -> (B, C)
            probs = F.softmax(logits, dim=-1)  # scores -> probabilities
            next_idx = torch.multinomial(probs, num_samples=1)  # sample 1 char
            idx = torch.cat((idx, next_idx), dim=1)  # append it
        return idx

model = BigramLanguageModel(vocab_size).to(device)

# ----------------------------------------------------------------------------
# Loss estimate helper (averaged over several batches = less noisy)
# ----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(200)
        for k in range(200):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ----------------------------------------------------------------------------
# Before training: generate from the untrained model (should be gibberish)
# ----------------------------------------------------------------------------
start = torch.zeros((1, 1), dtype=torch.long, device=device)  # a single \n token
print("\n--- BEFORE training (random weights) ---")
print(decode(model.generate(start, max_new_tokens=200)[0].tolist()))

# ----------------------------------------------------------------------------
# The training loop
# ----------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n--- Training ---")
for step in range(max_steps + 1):
    if step % 500 == 0:
        losses = estimate_loss()
        print(f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")        # 1) get a batch
    logits, loss = model(xb, yb)       # 2) forward pass -> loss
    optimizer.zero_grad(set_to_none=True)  # 3) reset old gradients
    loss.backward()                    # 4) backward pass: autograd fills gradients
    optimizer.step()                   # 5) nudge weights to reduce the loss

# ----------------------------------------------------------------------------
# After training: generate again
# ----------------------------------------------------------------------------
print("\n--- AFTER training ---")
print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))
