# Lesson 1: Tokenization
# Goal: chop text into pieces and convert to numbers
# This is the very first step in how any language model works.

text = "hello how are you i am a robot hello i am fine how are you"

# --- Step 1: Build the vocabulary ---
# A "vocabulary" is just the list of unique characters the model knows about.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# --- Step 2: Create lookup tables ---
# char_to_num: given a character, what number is it?
# num_to_char: given a number, what character is it?
char_to_num = {}
for i, ch in enumerate(chars):
    char_to_num[ch] = i

num_to_char = {}
for i, ch in enumerate(chars):
    num_to_char[i] = ch

# --- Step 3: Encode and decode ---
# Encoding: turn text into a list of numbers (what the model reads)
encoded = [char_to_num[ch] for ch in text]

# Decoding: turn numbers back into text (what we read)
decoded = "".join(num_to_char[num] for num in encoded)

# --- Step 4: Print results ---
print(f"Original text : {text}")
print(f"Vocab size    : {vocab_size}")
print(f"Vocabulary    : {chars}")
print()
print(f"Encoded       : {encoded}")
print()
print(f"Decoded       : {decoded}")
print()
print(f"Round-trip OK : {text == decoded}")

# --- Bonus: show the lookup table so it's clear what each number means ---
print()
print("Character → Number map:")
for ch, num in sorted(char_to_num.items(), key=lambda x: x[1]):
    label = repr(ch)   # shows space as ' ' clearly
    print(f"  {label:4s} → {num}")
