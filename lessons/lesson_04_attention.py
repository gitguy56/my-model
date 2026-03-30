# =============================================================================
# LESSON 4: Attention
# =============================================================================
# What we'll build:
#   - Self-attention from scratch (the heart of a Transformer)
#
# The idea:
#   When the model reads "how are you", each word looks at every other
#   word and decides how much to "pay attention" to it.
#   "you" pays a lot of attention to "are" and "how" — less to random words.
#
# Steps:
#   1. Query, Key, Value projections (Q, K, V)
#   2. Attention scores  →  scores = Q @ K.T / sqrt(d_k)
#   3. Masking           →  can't look at future tokens (causal mask)
#   4. Softmax           →  scores become probabilities (sum to 1)
#   5. Weighted sum      →  output = scores @ V
#
# Concepts covered:
#   - scaled dot-product attention
#   - causal (autoregressive) masking
#   - multi-head attention
#
# Requires: pip install torch
# =============================================================================

# TODO: coming in Lesson 4
