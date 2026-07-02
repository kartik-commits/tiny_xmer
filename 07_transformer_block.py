"""
Part 4: The full Transformer block.

We assemble four pieces on top of the single attention head from Part 3:
  1. Multi-head attention  - several heads in parallel, then a projection
  2. Feed-forward network  - per-token "thinking" after communication
  3. Residual connections  - add input back to each sublayer's output
  4. Layer norm            - normalize activations so deep stacks train stably

A "Block" = communication (attention) followed by computation (feed-forward),
each wrapped with a residual connection and pre-layer-norm.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# Config for this demo
B, T, C = 4, 8, 32     # batch, time, embedding dimension
n_head = 4             # number of attention heads
block_size = T


# ---------------------------------------------------------------------------
# One head of self-attention (from Part 3, now as a reusable module)
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        # a non-learnable buffer for the causal mask
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v                                        # (B,T,head_size)


# ---------------------------------------------------------------------------
# Piece 1: Multi-head attention = several heads in parallel + a projection
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, C)   # combine heads back to C

    def forward(self, x):
        # run every head, concatenate their outputs along the channel dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B,T,n_head*head_size)
        return self.proj(out)                                 # (B,T,C)


# ---------------------------------------------------------------------------
# Piece 2: Feed-forward network = per-token MLP (think after you communicate)
# The inner layer is 4x wider (from the paper); ReLU adds non-linearity.
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, 4 * C),
            nn.ReLU(),
            nn.Linear(4 * C, C),   # project back to C
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# The Block: pieces 1-4 together
#   x = x + attention(norm(x))     <- residual + pre-layernorm
#   x = x + feedforward(norm(x))   <- residual + pre-layernorm
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, C, n_head):
        super().__init__()
        head_size = C // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(C)
        self.ln1 = nn.LayerNorm(C)   # Piece 4: layer norm
        self.ln2 = nn.LayerNorm(C)

    def forward(self, x):
        # Piece 3: the "x +" is the residual connection (a gradient highway).
        x = x + self.sa(self.ln1(x))      # communicate
        x = x + self.ffwd(self.ln2(x))    # think
        return x


# ---------------------------------------------------------------------------
# Try it: a Block preserves shape (B,T,C) -> (B,T,C), so we can stack many.
# ---------------------------------------------------------------------------
x = torch.randn(B, T, C)
block = Block(C, n_head)
out = block(x)
print(f"input  shape: {tuple(x.shape)}")
print(f"output shape: {tuple(out.shape)}   (same shape -> blocks are stackable)")

n_params = sum(p.numel() for p in block.parameters())
print(f"parameters in one block: {n_params:,}")

# Stack 3 blocks to prove they compose
stack = nn.Sequential(*[Block(C, n_head) for _ in range(3)])
print(f"output of 3 stacked blocks: {tuple(stack(x).shape)}")
