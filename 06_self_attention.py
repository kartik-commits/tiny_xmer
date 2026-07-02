"""
Part 3 (Step 2): A single head of self-attention.

Now the attention weights are DATA-DEPENDENT. Each token produces:
  query (q): what am I looking for?
  key   (k): what do I contain?
  value (v): what I'll share if attended to
affinity(i, j) = q_i . k_j   -> mask future -> softmax -> weighted sum of values.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# Toy batch: B=4 sequences, T=8 positions, C=32 numbers per token (the
# "embedding dimension" — richer than the 2 we used for intuition).
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

# --- One head of self-attention ---
head_size = 16   # size of the query/key/value vectors

# Three separate linear layers project each token into q, k, v (no bias).
key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)      # (B, T, head_size)
q = query(x)    # (B, T, head_size)
v = value(x)    # (B, T, head_size)

# Affinities = every query dotted with every key.
# q @ k transposed over the last two dims -> (B, T, T): for each token, a score
# against every other token. Scale by 1/sqrt(head_size) to keep variance ~1 so
# softmax doesn't get too "peaky" early on.
affinities = q @ k.transpose(-2, -1) * head_size ** -0.5   # (B, T, T)

# Causal mask: token t may not attend to tokens > t (the future).
tril = torch.tril(torch.ones(T, T))
affinities = affinities.masked_fill(tril == 0, float("-inf"))

# Softmax each row -> attention weights that sum to 1.
weights = F.softmax(affinities, dim=-1)   # (B, T, T)

# Output = weighted sum of VALUES (not the raw x).
out = weights @ v   # (B, T, head_size)

print(f"input  x shape: {tuple(x.shape)}")
print(f"output   shape: {tuple(out.shape)}   (B, T, head_size)")
print("\nAttention weights for sequence 0 (row t = how token t distributes")
print("its attention over tokens 0..t). Note they are NO LONGER uniform:")
torch.set_printoptions(precision=3, sci_mode=False)
print(weights[0])
