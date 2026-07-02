# Building a Transformer from Scratch — A Beginner's Guide

This is a noob-friendly hands-on tutorial for building a **tiny char-level GPT** (a small
version of the model behind ChatGPT) from the ground up in PyTorch. It's written
for someone comfortable with Python but **new to machine learning** — every
concept is explained in plain terms.

By the end, we'll have a model that reads Shakespeare and generates new
Shakespeare-like text, one character at a time — and we'll understand *why*
every piece exists.

---

## How to use this guide

- Each **Part** introduces one concept, with a runnable script (`01_*.py`,
  `02_*.py`, …).
- Run each script yourself, read its output, and read the matching section here.
- The scripts are heavily commented — the code and this doc are meant to be read
  side by side.

---

## The big picture: what is a language model?

A language model does exactly one thing: **predict the next token given the
tokens so far.** A "token" is just a small unit of text — for us, a single
character.

Show it `"To be or not to b"` and it predicts the next character is `"e"`. Do
that repeatedly — feed each prediction back in — and it *generates* text. That's
it. Everything else (attention, transformer blocks, GPT) is machinery to make
that one prediction as accurate as possible.

We won't hand-write any calculus. PyTorch's **autograd** computes all the
derivatives needed for learning automatically. Our job is to define the math of
the model; PyTorch figures out how to improve it.

---

## Roadmap

| Part | Topic | Status |
|------|-------|--------|
| 1 | Data & tokenization — text → numbers | Done |
| 2 | Batching + a bigram baseline + the training loop | Done |
| 3 | Self-attention, built up slowly (the core idea) | Done |
| 4 | A full Transformer block (multi-head attn, FFN, residuals, layer norm) | Done |
| 5 | Stack blocks into a GPT + positional embeddings, train it | Done |
| 6 | Sample text, tune, visualize attention, connect to the paper | Done |
| 7 | Scale up: a bigger, longer-trained model | Done |

---

## Setup

- **Language:** Python 3.14
- **Framework:** PyTorch 2.12 (`pip install torch`)
- **Hardware:** a CUDA GPU is available here, so training will be fast (a CPU
  works too, just slower).
- **Dataset:** `input.txt` — the "Tiny Shakespeare" file (~1.1M characters), all
  of Shakespeare's works concatenated into one text file.

Verify your setup:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Part 1 — Data & Tokenization

**Files:** `01_tokenizer.py`, `02_data.py`

### Key idea

A neural network only does math on numbers — it can't read letters. So step one
is **tokenization**: choose a vocabulary and map each token to an integer.

We use the simplest scheme: **character-level tokenization**. Each unique
character is one token. Big models like GPT-4 use *subword* tokens (chunks like
`"ing"`) and a vocabulary of ~100,000; ours is just the **65 unique characters**
in Shakespeare. Smaller vocab = simpler model = easier to learn from.

### The tokenizer (`01_tokenizer.py`)

1. Read the text file.
2. Find the **vocabulary** — the sorted set of unique characters.
3. Build two lookup tables:
   - `stoi` ("string to integer"): `'a'` → `39`
   - `itos` ("integer to string"): `39` → `'a'`
4. `encode(text)` → list of ints; `decode(ints)` → text.

The same character always maps to the same integer — that consistency is what
lets the model learn patterns. Encoding then decoding returns the original text
exactly (no information lost).

```
Vocabulary size: 65
Characters: "\n !$&',-.3:;?ABCD…xyz"
encode("Hello there!") -> [20, 43, 50, 50, 53, 1, 58, 46, 43, 56, 43, 2]
                                      ^^  ^^  the two 50s are the two 'l's
```

### The dataset as a tensor (`02_data.py`)

- A **tensor** is PyTorch's n-dimensional array — like a NumPy array, but it can
  live on the GPU and supports automatic differentiation.
- We encode the *entire* dataset into one long 1-D tensor of 1,115,394 integers.
- **Train/validation split:** we keep the first 90% for training and hold back
  the last 10% as a *validation set*. We never train on the val set — it's how we
  check whether the model is learning **general patterns** rather than just
  memorizing the training text. Building this habit early is important.

```
data shape: (1115394,)   dtype: int64
Train set: 1,003,854 tokens
Val set:     111,540 tokens
```

### What you learned in Part 1

- Models see integers, not characters.
- Tokenization is the reversible map between text and integers.
- The whole corpus becomes one integer tensor, split into train/val.

---

## Part 2 — Batching + Bigram Baseline

**Files:** `03_batching.py`, `04_bigram.py`

### Step 1: Batching — the `(batch, time)` tensor

We never feed the whole 1.1M-character corpus at once. We take small **chunks**.

**Context length (`block_size`):** the maximum number of previous tokens the
model can see when predicting the next one. We use `block_size = 8` for now.

**One chunk = many examples.** For a chunk, the *input* `x` is characters
`0..7` and the *target* `y` is the same sequence shifted by one — characters
`1..8`. So the target at every position is simply "the next character." A chunk
of length 8 therefore contains 8 training examples at once:

```
given [18]                      -> predict 47
given [18, 47]                  -> predict 56
given [18, 47, 56]              -> predict 57
...
given [18, 47, 56, 57, 58, 1, 15, 47] -> predict 58
```

This teaches the model to handle contexts of *every* length from 1 up to
`block_size` — which is exactly what we need when generating text from a cold
start.

**Batching.** For speed (especially on a GPU) we process many chunks at once. We
pick `batch_size` random starting positions and stack the chunks into rows,
giving a 2-D tensor of shape **`(batch_size, block_size)`** — universally written
**`(B, T)`** for *(batch, time)*:

- `B` = batch = how many independent chunks we process together (they don't
  interact).
- `T` = time = the context length; positions left-to-right within a sequence.

```
inputs  shape: (4, 8)   # (batch, time)
targets shape: (4, 8)   # same shape, shifted left by one
```

`(B, T)` is the exact input shape that every transformer consumes, all the way
up to the real GPT. `get_batch()` in `03_batching.py` returns a fresh random
`(x, y)` pair each call — that's how we'll feed the training loop.

### Step 2: Bigram baseline + training loop

**The model.** A **bigram** predicts the next character from *only the current
character* — no attention, no memory of earlier context. In PyTorch it's a single
`nn.Embedding(vocab_size, vocab_size)` table: row `i` holds the scores
("**logits**") for what comes after token `i`. A forward pass on a `(B, T)` batch
returns `(B, T, vocab_size)` — a score for every possible next char at every
position.

**Loss = cross-entropy.** Measures how surprised the model is by the true next
character (lower = better). Reference point: random guessing over 65 classes is
`ln(65) ≈ 4.17`. `F.cross_entropy` expects `(N, C)` logits and `(N,)` targets, so
we flatten the batch and time dimensions together first.

**The training loop — the 5 steps every model here reuses:**

```python
xb, yb = get_batch("train")             # 1) grab a batch
logits, loss = model(xb, yb)            # 2) forward pass: predict + measure loss
optimizer.zero_grad(set_to_none=True)   # 3) clear old gradients
loss.backward()                         # 4) backward pass: autograd fills gradients
optimizer.step()                        # 5) nudge weights to reduce the loss
```

Repeat thousands of times. We use the **AdamW** optimizer (a robust default) and
periodically call `estimate_loss()`, which averages the loss over many batches
for a stable train/val readout.

**Generation.** To generate, we repeatedly: forward pass → take the logits at the
*last* position → `softmax` them into probabilities → `torch.multinomial` samples
the next character → append and repeat. Sampling (not always taking the argmax)
is what makes the output varied.

**Result.** Loss fell `4.73 → ~2.47`. The output is still gibberish as English,
but it learned real structure — the `SPEAKER:` play format, spaces, word-like
chunks:

```
BEFORE:  pYCXxfRkRZd wc'wfNfT;OLlTEeC K jxqPToTb?bXAUG:C   (pure noise)
AFTER:   BE:
         Wileranousel lind me l.
         HAshe ce hiry:                                    (structure!)
```

### Why the bigram is stuck — and what fixes it

The bigram can *never* do better than ~2.47, because it looks at only **one**
character. To predict well you need earlier context ("q" is usually followed by
"u"; but a name after a newline depends on the whole line). The tokens need a way
to **look back at and share information with previous tokens**. That mechanism is
**self-attention** — Part 3, the heart of the transformer.

---

## Part 3 — Self-Attention

**Files:** `05_attention_intuition.py`, `06_self_attention.py`

**Goal:** let each token gather information from itself and all **earlier**
tokens (never future ones — seeing the future would be cheating when the job is
to predict the next token).

### Step 1: The math trick — from averaging to a masked-softmax weighted sum

We build up "a token looks at the past" starting from the crudest version:
**just average the current token with all earlier ones.** Shown three ways that
give *identical* results:

1. **For-loop:** for each position `t`, take `x[0..t].mean()`. Plain but slow.
2. **Matrix multiply:** a **weighted sum is a matrix multiply**. A
   lower-triangular matrix (`1`s on/below the diagonal, `0`s above), with each row
   normalized to sum to 1, computes all the averages at once: `weights @ x`. The
   `0`s in the upper triangle are what **block the future**.
3. **Masked softmax** — how real attention is written. Start from an all-zero
   `affinities` matrix, `masked_fill` future positions with `-inf`, then
   `softmax` each row. Because `softmax(-inf) = 0` and equal values share weight
   equally, this reproduces the exact same averaging matrix.

```
weights (lower-triangular, rows sum to 1):
[1.00 0    0    0   ...]
[0.50 0.50 0    0   ...]
[0.33 0.33 0.33 0   ...]
 ... token t attends equally to tokens 0..t, not at all to the future
```

**Why bother with the softmax form?** Right now the affinities are fixed at `0`,
so every past token gets *equal* weight (a boring average). The breakthrough:
make the affinities **data-dependent** — let each token *compute* how much it
cares about every other token based on their content. Same mask + softmax
structure; the weights stop being uniform. **That is self-attention** (Step 2).

### Step 2: Query, Key, Value — real self-attention

We make the affinities **data-dependent**. Each token is linearly projected into
three vectors:

- **Query (q)** — "what am I looking for?"
- **Key (k)** — "what do I contain / offer?"
- **Value (v)** — "what I'll actually share if attended to"

The full recipe (**scaled dot-product attention**):

```python
k = key(x); q = query(x); v = value(x)          # each (B, T, head_size)
affinities = q @ k.transpose(-2, -1) * head_size**-0.5   # (B, T, T) scores
affinities = affinities.masked_fill(tril == 0, float("-inf"))  # block the future
weights = F.softmax(affinities, dim=-1)         # (B, T, T) rows sum to 1
out = weights @ v                               # (B, T, head_size) weighted sum of VALUES
```

Reading it: **affinity(i, j) = query_i · key_j** — how much token *i* cares about
token *j*. Mask the future, softmax into weights, then output the weighted sum of
**values** (not the raw `x` — q/k decide *how much* to attend, v decides *what
flows*).

Result: the attention weights are now **non-uniform and content-driven** —
compare to Step 1's flat averaging matrix:

```
Step 1 (fixed average)      Step 2 (learned, data-dependent)
[0.33 0.33 0.33 ...]   ->   [0.307 0.289 0.404 ...]
```

Different input produces a different attention pattern. Still fully causal (future
= 0, rows sum to 1).

**Three things to remember:**
- Output aggregates **values**, not the input directly.
- The `* head_size ** -0.5` **scaling** keeps pre-softmax numbers moderate so
  softmax doesn't become near-one-hot too early (the "scaled" in scaled
  dot-product attention).
- It's called **self**-attention because q, k, v all come from the *same* `x`.
  (If q came from a different source — e.g. a decoder attending to an encoder —
  it'd be **cross**-attention.)

### What's still missing (→ Part 4)

One attention head captures *one* kind of relationship. Real transformers use
**multiple heads in parallel** (multi-head attention), plus a **feed-forward
network** (per-token "thinking"), **residual connections**, and **layer norm** to
train deep stacks stably. That's the full Transformer block — Part 4.

## Part 4 — The Full Transformer Block

**File:** `07_transformer_block.py`

We wrap the Part-3 attention head into a complete, stackable transformer layer by
adding four pieces. A **Block = communicate (attention) → think (feed-forward)**,
each sublayer wrapped in a residual connection with pre-layer-norm.

### Piece 1: Multi-head attention

One head learns *one* kind of relationship. Real language has many at once, so we
run **several heads in parallel**, each with its own q/k/v, concatenate their
outputs, and pass through a final linear **projection** back to `C`.

Sizing: with embedding `C = 32` and `n_head = 4`, each head is `32/4 = 8`-dim.
Concatenating `4 × 8 = 32` restores the original width, so shapes stay consistent.

### Piece 2: Feed-forward network (FFN)

After attention lets tokens *communicate*, a small per-token MLP lets each token
*think* about what it gathered: `Linear(C, 4C) → ReLU → Linear(4C, C)`.
- The **4× wider** middle layer (from the paper) gives room to compute.
- **ReLU** is the non-linearity — without it the whole model collapses to one
  linear function no matter how many layers.

### Piece 3: Residual connections

The `x = x + sublayer(x)` pattern: each sublayer *adds to* `x` instead of
replacing it. This creates a **gradient highway** — during backprop gradients flow
straight through the `+` unchanged — which is what makes deep stacks trainable.

### Piece 4: Layer norm

`nn.LayerNorm(C)` normalizes each token's `C` numbers to mean 0 / variance 1,
keeping activations in a healthy range so training stays stable. We apply it
**before** each sublayer (`sa(ln1(x))`) — the modern **pre-norm** arrangement,
which trains more reliably than the paper's original post-norm.

### Putting it together

```python
def forward(self, x):
    x = x + self.sa(self.ln1(x))     # communicate (residual + pre-norm)
    x = x + self.ffwd(self.ln2(x))   # think       (residual + pre-norm)
    return x
```

A Block maps `(B,T,C) → (B,T,C)` — **shape-preserving**, so blocks stack to any
depth. One block here has ~12,600 parameters; we stacked 3 as a sanity check.

## Part 5 — The Full GPT

**File:** `08_gpt.py` (sample saved to `sample_output.txt`)

We assemble every piece into a working GPT and train it on the GPU.

### The last missing idea: positional embeddings

Self-attention is **position-blind** — it treats the input as a *set*, so
`"the cat"` and `"cat the"` would look identical. To fix this we add a second
embedding table indexed by **position** (0…block_size−1). Each token's input
becomes:

```python
x = token_embedding(idx) + position_embedding(arange(T))
#   what the token IS    +  where the token SITS
```

Both tables are learned. Now the model knows both identity *and* order.

### Full architecture

```
idx (B,T)
  -> token_emb + pos_emb        (B,T,C)
  -> Block x n_layer            (B,T,C)   # stacked transformer blocks (Part 4)
  -> final LayerNorm            (B,T,C)
  -> Linear head                (B,T,vocab_size)  # logits for next char
```

### Other additions

- **Dropout** (`0.2`): randomly zeroes activations during training to prevent
  overfitting. Active in `.train()`, disabled in `.eval()`.
- **`generate()` crops context** to the last `block_size` tokens — the position
  table only knows positions `0…block_size−1`, so we can't feed more.

### Config used

```
block_size=128, n_embd=192, n_head=6, n_layer=6, dropout=0.2
batch_size=64, lr=3e-4, max_iters=3000
```

### Results

A **2.72 M-parameter** model trained on GPU in a couple of minutes:

```
iter    0 | train 4.3580 | val 4.3523
iter 1000 | train 1.7949 | val 1.9174
iter 2000 | train 1.5366 | val 1.7278
iter 3000 | train 1.4278 | val 1.6301
```

Val loss **1.63** — far below the bigram's floor of ~2.47, because attention lets
the model use real context instead of a single previous character. The small
train↔val gap (1.43 vs 1.63) shows dropout is keeping overfitting in check.

Sample output — correct play format, invented speakers (`CORIOLANUS`,
`KING RICHARD III`), English rhythm and vocabulary; not every word is real, as
expected for a small char-level model:

```
LUCIO:
Now lo grong's draner.

GLOUCESTER:
Walcome your trace forew y lords:
Honour the Remour swear youngle so, my lorr,
```

**The whole journey:** random noise → bigram's `SPEAKER:` skeleton (loss 2.47) →
GPT reading like Shakespeare through frosted glass (loss 1.63).

---

## Part 6 — Sampling, Visualization & the Paper

**Files:** `gpt_model.py` (reusable model), `09_sampling.py`,
`10_visualize_attention.py`. No new architecture — this part is about *using* and
*understanding* the trained model. `08_gpt.py` now also saves a checkpoint
(`ckpt.pt`) that these scripts load, so we don't retrain.

### Step 1: Better sampling — temperature & top-k (`09_sampling.py`)

Two knobs control the creativity/coherence trade-off when generating:

- **temperature** — divide the logits before softmax.
  `< 1` sharpens the distribution (safer, more repetitive); `> 1` flattens it
  (more creative, more typos). Observed:

  ```
  temp 0.5:  coherent but repetitive ("the ... the ... the")
  temp 1.0:  natural balance
  temp 1.4:  wild — "Puary, but all Clifingbring, you caith."
  ```

- **top-k** — before sampling, keep only the `k` most likely next tokens and set
  the rest to `-inf`. Stops rare, low-probability characters from derailing the
  text. `top_k=10` gives natural but cleaner output.

The same script also shows **prompted generation** — seed the model with your own
text (e.g. `"ROMEO:"`) and it continues from there.

### Step 2: Visualizing attention (`10_visualize_attention.py`)

Each `Head` now stashes its attention matrix (`self.att`). We push a prompt
through the model and plot those matrices as heatmaps (saved to `attention.png`).
Cell `(row i, col j)` = how much token `i` attended to token `j`.

What the picture reveals for `"ROMEO: But soft,"`:

- **Every plot is lower-triangular** — the upper-right is empty because the causal
  mask forbids attending to the future. Our theory made visible.
- **Different heads learned different jobs.** Some (e.g. Layer 0, Head 1) show a
  strong off-diagonal — each token attending to the *previous* token (local,
  bigram-like). Others (deeper layers) attend to specific earlier tokens like the
  word-start `"B"` or the `":"`, i.e. longer-range structure.
- This is concrete proof that **multi-head attention specializes** — exactly the
  claim from Part 4, now observable.

### Step 3: How this maps to "Attention Is All You Need" (Vaswani et al., 2017)

| Paper concept | Where it lives in our code |
|---------------|----------------------------|
| Scaled dot-product attention | `Head` (`q@k.T * head_size**-0.5` → mask → softmax → `@v`) |
| Multi-head attention | `MultiHeadAttention` (parallel `Head`s + `proj`) |
| Position-wise feed-forward | `FeedForward` (Linear → ReLU → Linear, 4× inner) |
| Add & Norm | residual `x + sublayer(...)` + `LayerNorm` (we use pre-norm) |
| Positional encoding | `position_embedding_table` (we *learn* it; paper used sinusoids) |
| Masked (causal) attention | the lower-triangular `tril` mask in `Head` |

**Key difference:** the paper describes an **encoder–decoder** for translation.
We built a **decoder-only** model (the GPT family) — just the masked-attention
stack, which is all you need for pure text generation. No encoder, no
cross-attention.

**You've now built and understood a complete GPT — end to end.**

---

## Part 7 — Scaling Up

**File:** `11_train_bigger.py` (checkpoint `ckpt_big.pt`, sample `sample_big.txt`)

The Part 5 model's output was recognizably Shakespeare-*shaped* but full of
invented words. That's expected — it's a small, briefly-trained, **character-level**
model. Part 7 pulls the most direct lever: **make the model bigger and train it
longer** (same architecture, reused from `gpt_model.py`).

### What changed

| Hyperparameter | Part 5 | Part 7 |
|----------------|--------|--------|
| embedding dim `n_embd` | 192 | **384** |
| context `block_size` | 128 | **192** |
| iterations | 3000 | **5000** |
| parameters | 2.72 M | **10.76 M** |

Sized to fit a 4GB GPU (GTX 1650): `batch_size=32`, and **best-checkpoint saving**
every 250 iters (keep the lowest-val model even if training stops early).

### Result

Val loss improved **1.63 → 1.497**. Each ~0.1 drop is a visible quality jump:

```
Part 5 (loss 1.63):  Why, An, sit hath man instrable Vorgelia.
Part 7 (loss 1.50):  KING EDWARD IV:
                     That, and my might broad freeds and thence be she,
                     ...
                     WARWICK:
                     Not this news, noble counsel of heaven cure!
```

Far more of it is now **real English** ("noble counsel of heaven", "the senators
of", "the heir body"). Some invented words remain (`freeds`, `sert`) — that's the
**character-level ceiling**.

### Knowing when to stop

Training ended at **train 1.19 vs val 1.51** — the widening gap signals the model
is starting to **overfit** (memorizing training text). More iterations wouldn't
help much here. The next real gains require a *different* lever, not just more of
the same:
- **Subword / BPE tokenization** → the model predicts word-pieces, so output is
  real words by construction (the biggest visual win).
- **More data** or a **learning-rate schedule** (warmup + decay).s

---

## FAQs for curiosity

**Q: Your model made up words like "freeds" — why, and how would you fix it?**
It's character-level, so it generates letter sequences that are statistically
plausible but not guaranteed to be real words. The fix is subword (BPE)
tokenization: with a vocabulary of real word-pieces, every output is a valid
piece, so made-up spellings largely disappear.

**Q: How did you know the model was starting to overfit?**
The train loss kept dropping (1.19) while validation loss flattened and ticked up
(~1.50–1.51). A widening train/val gap means it's fitting the training data
specifically rather than learning general patterns.

**Q: What gave the bigger model lower loss — width, context, or training length?**
All three contribute: more embedding width = more capacity per token; longer
context = more information to condition on; more iterations = more optimization.
Width and context raise the ceiling; training length approaches it.

**Q: Why save the *best* checkpoint instead of the final one?**
Because the final model can be slightly worse than an earlier one once overfitting
begins. Tracking the lowest validation loss and saving only on improvement ("early
stopping" in spirit) keeps the best-generalizing weights.

**Q: What is "temperature" in sampling?**
A divisor applied to the logits before softmax. Low temperature makes the model
more confident/deterministic (picks likely tokens); high temperature makes it more
random/creative. It trades coherence against diversity.

**Q: What does top-k sampling do and why use it?**
It restricts sampling to the `k` highest-probability tokens, zeroing the rest.
This prevents the long tail of unlikely tokens from occasionally being picked and
derailing the output, while still allowing variety among sensible choices.

**Q: How can you tell attention actually learned something meaningful?**
By visualizing the attention weights: they're causal (lower-triangular) and
different heads form distinct, interpretable patterns — attending to the previous
token, to word starts, to punctuation — rather than uniform noise.

**Q: Decoder-only vs the original encoder-decoder transformer — what's the difference?**
The original paper targets translation: an encoder reads the source, a decoder
attends to it (cross-attention) while generating the target. GPT is decoder-only —
one masked self-attention stack that predicts the next token. Simpler, and
sufficient for text generation.

**Q: Why save a checkpoint, and what's in it?**
So you can reuse a trained model without retraining. `ckpt.pt` stores the learned
weights (`state_dict`), the config needed to rebuild the architecture, and the
tokenizer maps (`stoi`/`itos`) so encoding/decoding stays consistent.

**Q: Why do transformers need positional embeddings?**
Self-attention is permutation-invariant — it sees the input as a set, so without
position info "the cat" and "cat the" are identical. Adding a learned embedding
per position injects order. (The original paper used fixed sinusoids; GPT-style
models learn them.)

**Q: Token vs positional embedding — what's the difference?**
The token embedding encodes *what* a token is (its identity); the positional
embedding encodes *where* it is in the sequence. They're summed so each input
vector carries both.

**Q: What does dropout do and when is it active?**
It randomly zeroes a fraction of activations during training, forcing redundancy
and reducing overfitting. It's on in `model.train()` and automatically off in
`model.eval()` (so evaluation/generation is deterministic w.r.t. the weights).

**Q: Why crop the context to block_size during generation?**
The positional embedding table only has rows for positions `0…block_size−1`.
Feeding a longer sequence would index a position it never learned, so we keep only
the most recent `block_size` tokens.

**Q: Your GPT hit 1.63 loss vs the bigram's 2.47 — why the big gap?**
The bigram predicts from only the current character; the GPT attends over up to
128 previous characters through stacked attention layers, so it captures spelling,
word boundaries, and the speaker/format structure the bigram can't see.

**Q: How would you improve the sample quality further?**
Scale up (larger `n_embd`, more layers/heads, longer `block_size`), train longer,
add a learning-rate schedule, or switch to subword tokenization so the model
predicts word-pieces instead of characters. More data/compute is the main lever.

**Q: Why multiple attention heads instead of one big one?**
Each head can specialize in a different relationship (syntax, punctuation,
long-range vs local) and they run in parallel at the same cost. One large head
would have to cram all patterns into a single attention distribution.

**Q: What does the feed-forward network add that attention doesn't?**
Attention *moves information between* tokens but does little per-token computation.
The FFN processes each token independently, giving the model capacity to
transform what attention gathered. Communication vs. computation.

**Q: What problem do residual connections solve?**
Vanishing gradients in deep networks. The `x + f(x)` skip lets gradients flow
directly to earlier layers, so stacking many blocks still trains. They also let a
block learn a small *refinement* to `x` rather than a full replacement.

**Q: Why layer norm, and why "pre-norm" vs "post-norm"?**
Layer norm keeps activations at a stable scale so training doesn't diverge.
Applying it *before* each sublayer (pre-norm) keeps the residual path clean and
trains more reliably at depth than the original paper's post-norm.

**Q: Why is the FFN's hidden layer 4× wider?**
It's the ratio from "Attention Is All You Need." The wider inner layer gives the
non-linearity more room to represent complex per-token functions before
projecting back down.

**Q: Why must a block preserve the (B,T,C) shape?**
So blocks are interchangeable and stackable — the output of one is a valid input
to the next, letting us build arbitrary depth by repetition.

Short answers to common "wait, why?" questions.

**Q: What actually is an "embedding table"?**
A lookup table of learnable numbers, one row per token. `nn.Embedding(65, 65)`
is a 65×65 grid; feeding it token `i` returns row `i`. The rows start random and
are improved by training like any other weights.

**Q: What are "logits"?**
Raw, unbounded scores the model outputs — one per possible next token. They
aren't probabilities yet. `softmax` turns a vector of logits into probabilities
(all positive, summing to 1).

**Q: Why cross-entropy for the loss?**
It measures how much probability the model assigned to the *correct* next token —
low when the model is confident and right, high when it's confident and wrong.
It's the standard loss for "pick the right class out of N," which is exactly
next-token prediction. Reference: random guessing over 65 classes ≈ `ln(65) ≈ 4.17`.

**Q: When generating, why sample instead of always taking the highest-scoring char?**
Always taking the top char (argmax) makes output repetitive and often gets stuck
in loops. Sampling from the probability distribution gives varied, more natural
text. (Later, "temperature" lets you dial randomness up or down.)

**Q: What's the difference between train loss and val loss?**
Train loss is measured on data the model learns from; val loss on held-out data
it never trains on. If val loss stops improving while train loss keeps dropping,
the model is **overfitting** (memorizing instead of generalizing).

**Q: What do B, T, C mean? They're everywhere.**
Standard shorthand for tensor dimensions: **B** = batch (chunks processed at
once), **T** = time (positions in the sequence, up to `block_size`), **C** =
channels (the size of the vector at each position — e.g. vocab size, or later the
embedding dimension).

**Q: Why AdamW and not plain gradient descent?**
AdamW is an optimizer that adapts the step size per-parameter and is far more
forgiving of hyperparameters. It's the standard robust default for transformers.

**Q: What's a "hyperparameter"?**
A setting *you* choose (batch size, block size, learning rate, number of layers),
as opposed to a *parameter* (weight) the model learns during training.

---

## Glossary

- **Token** — the smallest unit of text the model handles. Here, one character.
- **Vocabulary** — the full set of distinct tokens (65 for us).
- **Tensor** — PyTorch's n-dimensional array; runs on CPU/GPU, supports autograd.
- **Encode / decode** — convert text ↔ integers.
- **Train / validation split** — data the model learns from vs. held-out data
  used only to measure genuine learning.
- **Autograd** — PyTorch's automatic computation of derivatives, so we never
  hand-write calculus.
- **Context length / block size** — how many previous tokens the model can see
  when predicting the next one.
