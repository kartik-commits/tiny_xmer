"""
Part 3 (Step 1): The math trick behind self-attention.

Goal: let each token gather info from itself + all EARLIER tokens (never future
ones — that would be cheating when predicting the next token).

We start with the crudest version of "gather": just AVERAGE the past tokens.
We show it 3 ways that give identical results, ending with the matrix form that
real attention uses.
"""
import torch
from torch.nn import functional as F

torch.manual_seed(1337)

# A toy batch: B=1 sequence, T=8 positions, C=2 numbers per position.
B, T, C = 1, 8, 2
x = torch.randn(B, T, C)
print("x (our input, 8 tokens each with 2 numbers):")
print(x[0])

# ---------------------------------------------------------------------------
# Version 1: obvious but slow — a Python for-loop averaging the past
# ---------------------------------------------------------------------------
xbow = torch.zeros(B, T, C)          # "bag of words" = averaged past
for b in range(B):
    for t in range(T):
        prev = x[b, :t + 1]          # all tokens from 0..t  -> (t+1, C)
        xbow[b, t] = prev.mean(dim=0)  # average them
print("\nVersion 1 (for-loop average). Row t = average of rows 0..t:")
print(xbow[0])

# ---------------------------------------------------------------------------
# Version 2: the matrix trick — a weighted sum IS a matrix multiply
# A lower-triangular matrix of averaging weights does the same thing at once.
# ---------------------------------------------------------------------------
weights = torch.tril(torch.ones(T, T))     # 1s on/below the diagonal, 0s above
weights = weights / weights.sum(dim=1, keepdim=True)  # each row sums to 1
print("\nThe averaging weight matrix (row t averages tokens 0..t):")
print(weights)
xbow2 = weights @ x                        # (T,T) @ (B,T,C) -> (B,T,C)
print("\nVersion 2 (matrix multiply) matches Version 1:",
      torch.allclose(xbow, xbow2))

# ---------------------------------------------------------------------------
# Version 3: the softmax form — how REAL attention writes it
# Start from "affinities" (how much each token attends to each other token).
# For plain averaging they're all 0; the mask forbids looking at the FUTURE.
# ---------------------------------------------------------------------------
tril = torch.tril(torch.ones(T, T))
affinities = torch.zeros(T, T)             # 0 = "no preference yet"
affinities = affinities.masked_fill(tril == 0, float("-inf"))  # block the future
weights3 = F.softmax(affinities, dim=-1)   # -inf -> 0 prob; rest share equally
xbow3 = weights3 @ x
print("\nVersion 3 (masked softmax) matches Version 1:",
      torch.allclose(xbow, xbow3))
print("\nThe softmax weights (identical to the averaging matrix):")
print(weights3)
