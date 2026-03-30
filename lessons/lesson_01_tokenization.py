# =============================================================================
# LESSON 1: Tokenization
# =============================================================================
# What is tokenization?
#   Before a language model can read text, it needs to convert that text
#   into numbers. That process is called tokenization.
#
# In this lesson we use the simplest possible approach:
#   character-level tokenization — every single character gets its own number.
#
# The pipeline:
#   text  →  list of characters  →  list of numbers  (this is encoding)
#   list of numbers  →  list of characters  →  text   (this is decoding)
# =============================================================================

# ---------------------------------------------------------
# 1. Our training text
# ---------------------------------------------------------
# In a real model this would be millions of words from books,
# websites, etc. We keep it tiny so we can see exactly what happens.

text = "hello how are you i am a robot hello i am fine how are you"

# ---------------------------------------------------------
# 2. Build the vocabulary
# ---------------------------------------------------------
# The vocabulary is the complete set of characters the model knows.
# sorted() + set() gives us unique characters in alphabetical order.

chars = sorted(set(text))
vocab_size = len(chars)

print("=" * 50)
print("STEP 1 — Vocabulary")
print("=" * 50)
print(f"Unique characters : {chars}")
print(f"Vocabulary size   : {vocab_size}")
# You'll see 16 characters: the space plus a-y (only letters that appear).
# A model trained on all of English would have ~70 characters.

# ---------------------------------------------------------
# 3. Create lookup tables
# ---------------------------------------------------------
# We need two dictionaries:
#   char_to_num  →  given a character, return its number  (used when encoding)
#   num_to_char  →  given a number, return its character  (used when decoding)

char_to_num = {ch: i for i, ch in enumerate(chars)}
num_to_char = {i: ch for i, ch in enumerate(chars)}

print()
print("=" * 50)
print("STEP 2 — Lookup tables")
print("=" * 50)
for ch, num in char_to_num.items():
    label = "SPACE" if ch == " " else f"  '{ch}'"
    print(f"  {label}  →  {num}")

# ---------------------------------------------------------
# 4. Encode: text → numbers
# ---------------------------------------------------------
# Walk through every character in the text and swap it for its number.
# This is the format the neural network will actually read.

encoded = [char_to_num[ch] for ch in text]

print()
print("=" * 50)
print("STEP 3 — Encode (text → numbers)")
print("=" * 50)
print(f"Original  : {text}")
print(f"Encoded   : {encoded}")

# ---------------------------------------------------------
# 5. Decode: numbers → text
# ---------------------------------------------------------
# The reverse: take a list of numbers and recover the original text.
# After training, this is how we'll turn the model's output back into words.

decoded = "".join(num_to_char[n] for n in encoded)

print()
print("=" * 50)
print("STEP 4 — Decode (numbers → text)")
print("=" * 50)
print(f"Decoded   : {decoded}")
print(f"Matches original? {text == decoded}")

# ---------------------------------------------------------
# 6. Helper functions (reusable in later lessons)
# ---------------------------------------------------------
# We wrap encode/decode into functions so future lessons can import them.

def encode(s):
    """Turn a string into a list of integers."""
    return [char_to_num[ch] for ch in s]

def decode(nums):
    """Turn a list of integers back into a string."""
    return "".join(num_to_char[n] for n in nums)

print()
print("=" * 50)
print("STEP 5 — Reusable encode/decode functions")
print("=" * 50)
test = "hi robot"
print(f"encode('{test}') → {encode(test)}")
print(f"decode back      → {decode(encode(test))}")

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print()
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Vocab size : {vocab_size} characters")
print(f"  Text length: {len(text)} characters  →  {len(encoded)} tokens")
print()
print("Next up — Lesson 2: Neural Network Basics")
print("  We'll turn these token numbers into 'embeddings'")
print("  (lists of floats the network can do math on)")
print("  and build our very first PyTorch layer.")
