"""
Part 6 (Step 2): Visualize attention.

We feed a short prompt through the trained model, grab the attention matrix from
one head (saved in Head.att), and plot it as a heatmap. Each cell (row i, col j)
= how much token i attended to token j. The upper triangle is empty because of
the causal mask (a token can't attend to the future).

Produces attention.png. Run after 08_gpt.py has produced ckpt.pt.
"""
import torch
import matplotlib
matplotlib.use("Agg")   # no display needed; save straight to a file
import matplotlib.pyplot as plt
from gpt_model import load_from_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
model, stoi, itos, cfg = load_from_checkpoint("ckpt.pt", device)

# A short prompt whose attention pattern we'll inspect.
prompt = "ROMEO: But soft,"
idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)

# One forward pass populates every Head.att.
with torch.no_grad():
    model(idx)

tokens = list(prompt)
n_layer = len(model.blocks)
n_head = len(model.blocks[0].sa.heads)
print(f"Model has {n_layer} layers x {n_head} heads. Prompt length: {len(tokens)}")

# Plot a grid: pick a few (layer, head) pairs to compare their patterns.
pairs = [(0, 0), (0, 1), (n_layer // 2, 0), (n_layer - 1, 0),
         (n_layer - 1, 1), (n_layer - 1, n_head - 1)]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (layer, head) in zip(axes.flat, pairs):
    att = model.blocks[layer].sa.heads[head].att[0].cpu().numpy()  # (T, T)
    im = ax.imshow(att, cmap="viridis")
    ax.set_title(f"Layer {layer}, Head {head}")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("attends to (key)")
    ax.set_ylabel("query token")
    fig.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('Attention weights for prompt: "ROMEO: But soft,"', fontsize=14)
fig.tight_layout()
fig.savefig("attention.png", dpi=120)
print("Saved heatmap to attention.png")

# Also print one head as numbers so it's readable without opening the image.
att0 = model.blocks[n_layer - 1].sa.heads[0].att[0].cpu()
torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)
print(f"\nAttention matrix, last layer head 0 (rows=query, cols=key):")
print("tokens:", tokens)
print(att0)
