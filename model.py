"""
model.py — GPT-style transformer, built from scratch.

Architecture (same family as GPT-2):
  token embedding + positional embedding
  → N × TransformerBlock (multi-head attention + feed-forward)
  → LayerNorm
  → linear head → logits over vocabulary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Single self-attention head."""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # causal mask: token at position i can only attend to positions ≤ i
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # attention scores, scaled
        scale = k.shape[-1] ** -0.5
        wei = q @ k.transpose(-2, -1) * scale   # (B, T, T)

        # mask future positions (causal / autoregressive)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)   # (B, T, head_size)
        return wei @ v      # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """N attention heads run in parallel, then projected back."""

    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        head_size = n_embd // n_heads
        self.heads   = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(n_heads)
        ])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network (expand → activate → contract)."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block:
      x = x + attention(layernorm(x))
      x = x + feedforward(layernorm(x))
    The residual connections (x + ...) let gradients flow back easily.
    """

    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_heads, block_size, dropout)
        self.ff   = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    GPT language model.

    Config keys:
        vocab_size  — number of unique tokens
        n_embd      — embedding dimension (width of the model)
        n_heads     — number of attention heads per block
        n_layers    — number of transformer blocks (depth of the model)
        block_size  — maximum context length in tokens
        dropout     — dropout rate (0.0 = off, 0.1 = light regularisation)
    """

    def __init__(self, vocab_size, n_embd, n_heads, n_layers, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying: share token embedding and output head weights
        # (standard GPT trick — reduces parameters, improves performance)
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        tok = self.tok_emb(idx)                                         # (B, T, n_embd)
        pos = self.pos_emb(torch.arange(T, device=idx.device))         # (T, n_embd)
        x   = tok + pos                                                 # (B, T, n_embd)
        x   = self.blocks(x)
        x   = self.ln_f(x)
        logits = self.head(x)                                           # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=40):
        """
        Autoregressively generate `max_new_tokens` new tokens.

        idx           — (1, T) tensor of starting token IDs
        temperature   — >1 = more random, <1 = more focused
        top_k         — only sample from the top-k most likely tokens
        """
        for _ in range(max_new_tokens):
            # crop to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # (1, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, next_tok], dim=1)

        return idx

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
