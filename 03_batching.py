"""
Part 2 (Step 1): Batching — how we chop the data into training examples.

Two ideas:
  - block_size: the context length. The max number of previous tokens the model
    sees when predicting the next one.
  - A single chunk of length block_size secretly contains block_size training
    examples (predict char 2 from char 1, char 3 from chars 1-2, ...).
  - We stack many chunks into a (batch, time) tensor for speed.
"""
import torch

# --- Rebuild tokenizer + data (same as Part 1) ---
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
train_data = data[:n]
val_data = data[n:]

# ----------------------------------------------------------------------------
# 1) What "8 examples hide in 1 chunk" means
# ----------------------------------------------------------------------------
block_size = 8
x = train_data[:block_size]        # the input:  characters 0..7
y = train_data[1:block_size + 1]   # the target: characters 1..8 (shifted by 1)

print("One chunk unpacked into examples:")
for t in range(block_size):
    context = x[:t + 1]            # everything up to and including position t
    target = y[t]                 # the very next character
    print(f"  given {context.tolist()}  -> predict {target.item()}")

# ----------------------------------------------------------------------------
# 2) A real batch: stack several random chunks into a (batch, time) tensor
# ----------------------------------------------------------------------------
torch.manual_seed(1337)  # makes the "random" picks reproducible
batch_size = 4           # how many independent chunks we process at once

def get_batch(split):
    d = train_data if split == "train" else val_data
    # pick batch_size random starting positions
    ix = torch.randint(len(d) - block_size, (batch_size,))
    # stack the chunks into rows -> shape (batch_size, block_size)
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch("train")
print(f"\ninputs  shape: {tuple(xb.shape)}  (batch, time)")
print(xb)
print(f"targets shape: {tuple(yb.shape)}")
print(yb)
