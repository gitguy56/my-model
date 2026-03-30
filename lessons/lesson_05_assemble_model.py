# =============================================================================
# LESSON 5: Assemble the Model
# =============================================================================
# What we'll build:
#   - Wire together all the pieces into a complete mini GPT-style model
#
# Architecture (each block repeated N times):
#
#   Token IDs
#       ↓
#   Embedding layer   (token ID → float vector)
#       ↓
#   + Positional encoding  (adds "where am I in the sequence?")
#       ↓
#   [ Transformer Block ] × N
#       ├── Multi-head self-attention
#       ├── Add & LayerNorm
#       ├── Feed-forward network (two Linear layers + ReLU)
#       └── Add & LayerNorm
#       ↓
#   Linear head  (float vector → logits over vocabulary)
#       ↓
#   Softmax  →  probability of each next character
#
# Target size: ~50–150M parameters (fine for a Raspberry Pi with quantization)
#
# Requires: pip install torch
# =============================================================================

# TODO: coming in Lesson 5
