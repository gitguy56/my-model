# =============================================================================
# LESSON 3: The Training Loop
# =============================================================================
# In Lesson 2 we built the plumbing: embedding → linear → probabilities.
# The model's guesses were random garbage.
#
# The training loop fixes that. It repeats three steps thousands of times:
#
#   1. FORWARD  — model makes a prediction
#   2. LOSS     — we measure how wrong it was  (cross-entropy)
#   3. BACKWARD — math figures out which weights caused the error
#   4. STEP     — every weight nudges slightly in the right direction
#
# After enough repetitions the model stops being random and starts learning
# the patterns in the text.
# =============================================================================

import torch
import torch.nn as nn

torch.manual_seed(42)   # makes random numbers reproducible — same output every run

# ── vocabulary (same as Lessons 1 & 2) ───────────────────────────────────────
text      = "hello how are you i am a robot hello i am fine how are you"
chars     = sorted(set(text))
vocab_size = len(chars)
char_to_num = {ch: i for i, ch in enumerate(chars)}
num_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [char_to_num[ch] for ch in s]
def decode(nums): return "".join(num_to_char[n] for n in nums)

# =============================================================================
# PART 1 — Prepare training data
# =============================================================================
# We turn the text into (input, target) pairs.
# For every character, the input is that character and the target is the NEXT one.
#
#   text :  h  e  l  l  o  ...
#   input:  h  e  l  l      (every char except the last)
#   target: e  l  l  o      (every char except the first)

all_tokens = torch.tensor(encode(text), dtype=torch.long)

inputs  = all_tokens[:-1]    # everything except the last token
targets = all_tokens[1:]     # everything except the first token

print("=" * 55)
print("PART 1 — Training data")
print("=" * 55)
print(f"Total tokens : {len(all_tokens)}")
print(f"Input pairs  : {len(inputs)}")
print()
print("First 8 (input → target) pairs:")
for i in range(8):
    inp = num_to_char[inputs[i].item()]
    tgt = num_to_char[targets[i].item()]
    print(f"  '{inp}'  →  '{tgt}'")

# =============================================================================
# PART 2 — Build the model
# =============================================================================
# Same structure as Lesson 2: Embedding → Linear.
# This is called a "bigram" model — it predicts the next character
# using only the current character (no memory of what came before).
# Attention (Lesson 4) will give it memory.

EMBED_DIM = 32    # slightly bigger than Lesson 2 — more capacity to learn

class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.linear    = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, token_ids):
        x = self.embedding(token_ids)   # [N, EMBED_DIM]
        x = self.linear(x)              # [N, vocab_size]  ← these are logits
        return x

model = model_before = BigramModel()

# count parameters
total_params = sum(p.numel() for p in model.parameters())
print()
print("=" * 55)
print("PART 2 — Model")
print("=" * 55)
print(f"Parameters: {total_params}")
print(f"  Embedding : {vocab_size} × {EMBED_DIM} = {vocab_size * EMBED_DIM}")
print(f"  Linear    : {EMBED_DIM} × {vocab_size} = {EMBED_DIM * vocab_size}")

# =============================================================================
# PART 3 — Loss function and optimizer
# =============================================================================
# LOSS — cross-entropy: measures how surprised the model was by the correct answer.
#   Perfect prediction → loss = 0
#   Completely random  → loss ≈ log(vocab_size) ≈ 2.77 for vocab_size=16
#
# OPTIMIZER — Adam: a smart version of "nudge every weight a little bit".
#   learning_rate controls how big each nudge is.
#   Too big → model overshoots and never learns.
#   Too small → model learns painfully slowly.

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# show what random loss looks like before any training
with torch.no_grad():
    logits = model(inputs)
    initial_loss = loss_fn(logits, targets).item()

print()
print("=" * 55)
print("PART 3 — Loss before training")
print("=" * 55)
print(f"Initial loss : {initial_loss:.4f}")
print(f"(random baseline ≈ {torch.log(torch.tensor(float(vocab_size))):.4f}  — pure guessing)")

# =============================================================================
# PART 4 — The training loop
# =============================================================================

STEPS = 2000

print()
print("=" * 55)
print(f"PART 4 — Training  ({STEPS} steps)")
print("=" * 55)
print(f"{'Step':>6}   {'Loss':>8}   Note")
print("-" * 40)

for step in range(STEPS):
    # 1. forward pass — model predicts
    logits = model(inputs)               # shape: [57, 16]

    # 2. compute loss — how wrong was it?
    loss = loss_fn(logits, targets)

    # 3. backward pass — compute gradients
    optimizer.zero_grad()                # clear old gradients first
    loss.backward()                      # calculate new gradients

    # 4. optimizer step — update weights
    optimizer.step()

    # print progress every 200 steps
    if step == 0 or (step + 1) % 200 == 0:
        note = ""
        if step == 0:          note = "← first step, still random"
        elif loss.item() < 1.0: note = "← getting good!"
        elif loss.item() < 1.5: note = "← learning fast"
        print(f"{step+1:>6}   {loss.item():>8.4f}   {note}")

print()
print(f"Final loss: {loss.item():.4f}  (started at {initial_loss:.4f})")

# =============================================================================
# PART 5 — See what the model learned
# =============================================================================
# Before training: probabilities were all ~6% (random)
# After training:  probabilities should cluster on the correct next characters

print()
print("=" * 55)
print("PART 5 — What the model learned")
print("=" * 55)

test_chars = ['h', 'e', 'l', 'i', 'a', ' ']
for ch in test_chars:
    token_id = torch.tensor([char_to_num[ch]])
    with torch.no_grad():
        logits = model(token_id)
        probs  = torch.softmax(logits[0], dim=0)

    top3 = torch.topk(probs, 3)
    top3_chars = [(num_to_char[i.item()], p.item()) for i, p in zip(top3.indices, top3.values)]
    guesses = "  ".join(f"'{c}' {p:.0%}" for c, p in top3_chars)
    print(f"  After '{ch}'  →  top guesses: {guesses}")

print()
print("  Compare with the actual text:")
print(f"  '{text[:30]}...'")

# =============================================================================
# PART 6 — Generate text
# =============================================================================
# Now we can make the model talk.
# We give it one starting character and let it keep predicting the next one.

print()
print("=" * 55)
print("PART 6 — Generate text")
print("=" * 55)

def generate(start_char, length=40, temperature=0.8):
    """
    Generate `length` characters starting from `start_char`.
    temperature: higher = more random, lower = more repetitive
    """
    result = [char_to_num[start_char]]
    for _ in range(length):
        token_id = torch.tensor([result[-1]])
        with torch.no_grad():
            logits = model(token_id)
            probs  = torch.softmax(logits[0] / temperature, dim=0)
        next_token = torch.multinomial(probs, num_samples=1).item()
        result.append(next_token)
    return decode(result)

for start in ['h', 'i', 'a']:
    output = generate(start, length=50)
    print(f"  Starting with '{start}':  {output}")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Loss dropped from {initial_loss:.2f}  →  {loss.item():.2f}")
print(f"  The model has learned short-range patterns in the text.")
print()
print("  Limitation of this bigram model:")
print("  It only looks at ONE character at a time.")
print("  It has no memory of what came before.")
print()
print("Next up — Lesson 4: Attention")
print("  We'll give the model memory so it can look at")
print("  the whole sequence before making each prediction.")
