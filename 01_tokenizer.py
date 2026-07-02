"""
Part 1: Data & Tokenization
Goal: understand our text data and turn it into integers a model can use.
"""

# 1. Read the raw text file
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Total number of characters in the dataset: {len(text)}")

# 2. Find the "vocabulary": the set of unique characters that appear.
#    sorted() just gives us a stable, predictable order.
chars = sorted(set(text))
vocab_size = len(chars)
print(f"Vocabulary size (number of unique chars): {vocab_size}")
print(f"The characters are: {''.join(chars)!r}")

# 3. Build two lookup tables:
#    stoi = "string to integer"  ('a' -> 39)
#    itos = "integer to string"  (39 -> 'a')
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 4. "encode" turns a string into a list of integers.
#    "decode" turns a list of integers back into a string.
def encode(s):
    return [stoi[c] for c in s]

def decode(nums):
    return "".join(itos[i] for i in nums)

# 5. Let's prove it round-trips: text -> numbers -> text
sample = "Hello there!"
encoded = encode(sample)
print(f"\nOriginal text : {sample!r}")
print(f"Encoded (ints): {encoded}")
print(f"Decoded back  : {decode(encoded)!r}")
