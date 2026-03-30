"""
train.py — download data, train the GPT, save checkpoints.

Usage:
    python train.py

The script will:
  1. Download the Tiny Shakespeare dataset (~1 MB of text) if not present
  2. Build the character-level vocabulary
  3. Train the model, printing loss every 500 steps
  4. Save a checkpoint to checkpoints/ckpt.pt when done

Adjust the CONFIG block below to change model size or training length.
"""

import os
import math
import time
import torch

from model import GPT

# =============================================================================
# CONFIG — change these to make the model bigger/smaller/faster/slower
# =============================================================================
CONFIG = dict(
    # ── model architecture ────────────────────────────────────────────────────
    n_embd     = 128,    # embedding dimension  (bigger = smarter, slower)
    n_heads    = 4,      # attention heads      (must divide n_embd evenly)
    n_layers   = 4,      # transformer blocks   (deeper = smarter, slower)
    block_size = 64,     # context window       (how many chars the model sees at once)
    dropout    = 0.1,

    # ── training ──────────────────────────────────────────────────────────────
    batch_size    = 16,     # sequences per training step
    max_steps     = 8000,   # total training steps  (raise to 20000+ for better quality)
    eval_interval = 100,    # print loss every N steps
    learning_rate = 3e-4,

    # ── data ──────────────────────────────────────────────────────────────────
    data_path  = "data/robot_chat.txt",
    train_split= 0.9,       # 90% train, 10% validation

    # ── output ────────────────────────────────────────────────────────────────
    checkpoint_dir = "checkpoints",
)

# =============================================================================
# 1. Get the data
# =============================================================================

def download_data():
    if not os.path.exists(CONFIG["data_path"]):
        raise FileNotFoundError(
            f"Training data not found at '{CONFIG['data_path']}'.\n"
            "Make sure data/robot_chat.txt exists."
        )
    print(f"Data: {CONFIG['data_path']}")

# =============================================================================
# 2. Build vocabulary and encode the text
# =============================================================================

def build_vocab(text):
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_num = {ch: i for i, ch in enumerate(chars)}
    num_to_char = {i: ch for i, ch in enumerate(chars)}
    return vocab_size, char_to_num, num_to_char

def encode(text, char_to_num):
    return torch.tensor([char_to_num[ch] for ch in text], dtype=torch.long)

# =============================================================================
# 3. Batch sampling
# =============================================================================

def get_batch(data, block_size, batch_size, device):
    """Sample a random batch of (input, target) sequence pairs."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i     : i + block_size] for i in ix])
    y  = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# =============================================================================
# 4. Evaluation
# =============================================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_steps=50):
    model.eval()
    results = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_steps):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results

# =============================================================================
# 5. Main training script
# =============================================================================

def main():
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    download_data()
    text = open(CONFIG["data_path"], encoding="utf-8").read()
    print(f"Dataset: {len(text):,} characters")

    vocab_size, char_to_num, num_to_char = build_vocab(text)
    print(f"Vocab size: {vocab_size} unique characters")

    # save vocab for generate.py to use later
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    torch.save({"char_to_num": char_to_num, "num_to_char": num_to_char},
               os.path.join(CONFIG["checkpoint_dir"], "vocab.pt"))

    data = encode(text, char_to_num).to(device)
    n    = int(CONFIG["train_split"] * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"Train tokens: {len(train_data):,}   Val tokens: {len(val_data):,}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = GPT(
        vocab_size = vocab_size,
        n_embd     = CONFIG["n_embd"],
        n_heads    = CONFIG["n_heads"],
        n_layers   = CONFIG["n_layers"],
        block_size = CONFIG["block_size"],
        dropout    = CONFIG["dropout"],
    ).to(device)

    print(f"\nModel parameters: {model.num_parameters():,}")
    print(f"  n_embd={CONFIG['n_embd']}, n_heads={CONFIG['n_heads']}, "
          f"n_layers={CONFIG['n_layers']}, block_size={CONFIG['block_size']}")

    # ── optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # cosine learning rate decay
    def get_lr(step):
        warmup = 200
        if step < warmup:
            return CONFIG["learning_rate"] * step / warmup
        progress = (step - warmup) / (CONFIG["max_steps"] - warmup)
        return CONFIG["learning_rate"] * 0.5 * (1 + math.cos(math.pi * progress))

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {CONFIG['max_steps']:,} steps...")
    print(f"{'Step':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Time':>8}  ETA")
    print("-" * 55)

    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(CONFIG["max_steps"]):
        # update learning rate
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # evaluate and print progress
        if step % CONFIG["eval_interval"] == 0 or step == CONFIG["max_steps"] - 1:
            losses = estimate_loss(model, train_data, val_data,
                                   CONFIG["block_size"], CONFIG["batch_size"], device)
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            remaining = (CONFIG["max_steps"] - step) / steps_per_sec if steps_per_sec > 0 else 0
            eta = f"{remaining/60:.0f}m" if remaining > 60 else f"{remaining:.0f}s"
            print(f"{step:>6}  {losses['train']:>10.4f}  {losses['val']:>10.4f}  {elapsed:>7.1f}s  ETA {eta}")

            # save best checkpoint
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "step":       step,
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "config":     CONFIG,
                    "val_loss":   best_val_loss,
                }, os.path.join(CONFIG["checkpoint_dir"], "ckpt.pt"))

        # training step
        x, y = get_batch(train_data, CONFIG["block_size"], CONFIG["batch_size"], device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to {CONFIG['checkpoint_dir']}/ckpt.pt")
    print("\nRun  python generate.py  to generate text.")


if __name__ == "__main__":
    main()
