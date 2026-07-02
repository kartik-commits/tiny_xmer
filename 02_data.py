"""
Part 1 (cont.): Encode the whole dataset into a tensor and split train/val.
"""
import torch

# --- Rebuild the tokenizer from Part 1 ---
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda nums: "".join(itos[i] for i in nums)

# --- Encode the ENTIRE text into one long 1-D tensor of integers ---
# torch.tensor(...) turns a Python list into a PyTorch tensor (an n-dim array
# that lives on CPU/GPU and supports fast math). dtype=long = 64-bit integers.
data = torch.tensor(encode(text), dtype=torch.long)
print(f"data tensor shape: {tuple(data.shape)}, dtype: {data.dtype}")
print(f"First 40 numbers: {data[:40].tolist()}")
print(f"...which decode to: {decode(data[:40].tolist())!r}")

# --- Train / validation split: first 90% to train on, last 10% held out ---
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"\nTrain set: {len(train_data)} tokens")
print(f"Val set  : {len(val_data)} tokens")
