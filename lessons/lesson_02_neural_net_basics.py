# =============================================================================
# LESSON 2: Neural Network Basics
# =============================================================================
# In Lesson 1 we turned text into numbers.
# Now we ask: what does the network actually DO with those numbers?
#
# Step 1 — Embedding : token number  →  list of floats (a "meaning vector")
# Step 2 — Linear    : list of floats →  scores for every character in vocab
# Step 3 — Softmax   : scores         →  probabilities (which char comes next?)
#
# The weights start random, so predictions will be terrible.
# That's fine — Lesson 3 (the training loop) will fix that.
# =============================================================================

import torch
import torch.nn as nn

# ── re-use the vocabulary we built in Lesson 1 ────────────────────────────────
text = "hello how are you i am a robot hello i am fine how are you"
chars = sorted(set(text))
vocab_size = len(chars)                  # 16
char_to_num = {ch: i for i, ch in enumerate(chars)}
num_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [char_to_num[ch] for ch in s]

def decode(nums):
    return "".join(num_to_char[n] for n in nums)

# =============================================================================
# PART 1 — Embeddings
# =============================================================================
# An embedding is a lookup table.
# Each token ID maps to a row of floats — its "meaning vector".
#
# EMBED_DIM = how many floats describe each character.
# We use 16 here (tiny). Real models use 512–4096.

EMBED_DIM = 16

embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBED_DIM)
# Shape of the table: [vocab_size × EMBED_DIM]  →  [16 × 16]
# Every row is one character. Values start random; training will adjust them.

print("=" * 55)
print("PART 1 — Embeddings")
print("=" * 55)
print(f"Embedding table shape: {list(embedding_table.weight.shape)}")
print(f"  (one row per character, {EMBED_DIM} floats per row)\n")

# Look up a single token ──────────────────────────────────────────────────────
token_id = torch.tensor(char_to_num['h'])        # 'h' → 5
embed_h = embedding_table(token_id)

print(f"Token 'h'  (id={token_id.item()})")
print(f"Embedding  : {[round(x,3) for x in embed_h.tolist()]}")
print(f"            ↑ {EMBED_DIM} floats that 'represent' the letter h")
print()

# Look up a whole sequence ────────────────────────────────────────────────────
word = "hello"
token_ids = torch.tensor(encode(word))           # [5, 3, 7, 7, 10]
embeddings = embedding_table(token_ids)          # shape: [5, 16]

print(f"Word '{word}'  →  token ids: {token_ids.tolist()}")
print(f"Embedding matrix shape: {list(embeddings.shape)}")
print(f"  ({len(word)} tokens × {EMBED_DIM} floats each)\n")

for i, ch in enumerate(word):
    row = [round(x, 2) for x in embeddings[i].tolist()]
    print(f"  '{ch}'  →  {row}")

# =============================================================================
# PART 2 — Linear Layer (the simplest neural network layer)
# =============================================================================
# A Linear layer does one thing:   output = input × W  +  b
#   W = weight matrix (learned during training)
#   b = bias vector  (also learned)
#
# We use it to go from EMBED_DIM floats  →  vocab_size scores (one per char).
# Those scores are called "logits".

print()
print("=" * 55)
print("PART 2 — Linear Layer  (embedding → logits)")
print("=" * 55)

linear = nn.Linear(in_features=EMBED_DIM, out_features=vocab_size)

# Feed the embedding of 'h' through the linear layer
logits_h = linear(embed_h)

print(f"Input  (embedding of 'h') : {EMBED_DIM} floats")
print(f"Output (logits)           : {vocab_size} scores  (one per character)")
print()
print("Raw logits (meaningless before training):")
for i, score in enumerate(logits_h.tolist()):
    print(f"  '{num_to_char[i]}'  →  {score:.4f}")

# =============================================================================
# PART 3 — Softmax  (logits → probabilities)
# =============================================================================
# Softmax squashes any list of numbers into probabilities that sum to 1.
# The character with the highest probability is the model's best guess
# for what comes next.

print()
print("=" * 55)
print("PART 3 — Softmax  (logits → probabilities)")
print("=" * 55)

probs = torch.softmax(logits_h, dim=0)

print("Probability of each character following 'h':")
print("(random weights → roughly equal ~6% each — not useful yet)\n")

for i, p in enumerate(probs.tolist()):
    bar = "█" * int(p * 200)
    print(f"  '{num_to_char[i]}'  {p:.1%}  {bar}")

best_guess_id   = torch.argmax(probs).item()
best_guess_char = num_to_char[best_guess_id]
print(f"\nModel's best guess for what follows 'h': '{best_guess_char}'")
print("(With random weights this is meaningless — training fixes it)")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 55)
print("SUMMARY")
print("=" * 55)
print("  Embedding table : maps each of 16 token IDs → 16 floats")
print("  Linear layer    : maps 16 floats → 16 logits (one per char)")
print("  Softmax         : turns logits into probabilities")
print()
print("  Full pipeline so far:")
print("  text → token IDs → embeddings → logits → probabilities")
print()
print("Next up — Lesson 3: The Training Loop")
print("  We'll measure how wrong the model is (loss),")
print("  then automatically adjust every weight to make it less wrong.")
print("  Run this 1000 times and the model starts to learn.")
