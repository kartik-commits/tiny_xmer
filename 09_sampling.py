"""
Part 6 (Step 1): Better sampling — temperature and top-k.

The basic generate() sampled directly from the model's probabilities. Two knobs
let us control the creativity/coherence trade-off:

  temperature: divides the logits before softmax.
     < 1.0  -> sharper distribution -> safer, more repetitive
     = 1.0  -> the model's raw distribution
     > 1.0  -> flatter distribution -> more random / creative (more typos)

  top_k: keep only the k most likely next tokens, zero out the rest.
     Prevents rare, low-probability characters from derailing the text.

Run after 08_gpt.py has produced ckpt.pt.
"""
import torch
from torch.nn import functional as F
from gpt_model import load_from_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
model, stoi, itos, cfg = load_from_checkpoint("ckpt.pt", device)
decode = lambda nums: "".join(itos[i] for i in nums)
block_size = cfg["block_size"]


@torch.no_grad()
def generate(prompt="\n", max_new_tokens=300, temperature=1.0, top_k=None):
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature        # temperature scaling
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)            # k largest logits
            logits[logits < v[:, [-1]]] = float("-inf")  # drop everything else
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)
    return decode(idx[0].tolist())


torch.manual_seed(0)
print("=" * 70)
print("temperature = 0.5  (conservative — safer, more repetitive)")
print("=" * 70)
print(generate(temperature=0.5, max_new_tokens=250))

torch.manual_seed(0)
print("\n" + "=" * 70)
print("temperature = 1.0  (the model's natural distribution)")
print("=" * 70)
print(generate(temperature=1.0, max_new_tokens=250))

torch.manual_seed(0)
print("\n" + "=" * 70)
print("temperature = 1.4  (wild — more creative, more typos)")
print("=" * 70)
print(generate(temperature=1.4, max_new_tokens=250))

torch.manual_seed(0)
print("\n" + "=" * 70)
print("temperature = 1.0, top_k = 10  (natural but no rare-char derailing)")
print("=" * 70)
print(generate(temperature=1.0, top_k=10, max_new_tokens=250))

# Prompted generation: seed the model with your own text
torch.manual_seed(0)
print("\n" + "=" * 70)
print('prompt = "ROMEO:" (top_k=10)')
print("=" * 70)
print(generate(prompt="ROMEO:", temperature=1.0, top_k=10, max_new_tokens=250))
