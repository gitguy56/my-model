"""
finetune.py — fine-tune distilgpt2 on robot conversation data.

distilgpt2 already knows English (trained on 40GB of text).
We just teach it how your robot talks — takes ~10 minutes on CPU.

Usage:
    python finetune.py
Then chat:
    python chat.py
"""

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = "distilgpt2"          # 82M params, fast, already speaks English
DATA_PATH  = "data/robot_chat.txt"
SAVE_PATH  = "checkpoints/robot_gpt"

CONFIG = dict(
    max_length  = 128,    # tokens per training chunk
    batch_size  = 2,
    steps       = 600,    # ~10 min on CPU — raise to 1500 for better quality
    lr          = 5e-5,
    print_every = 50,
)

# =============================================================================
# 1. Load model + tokenizer
# =============================================================================
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.train()

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {total_params:,}")
print(f"  This model already speaks English — we just teach it the robot personality.\n")

# =============================================================================
# 2. Prepare training data
# =============================================================================
print(f"Loading training data from {DATA_PATH}...")
text = open(DATA_PATH, encoding="utf-8").read()

# repeat the data a few times so the model sees more examples per step
text = text * 6
print(f"  Training text: {len(text):,} characters")

encodings = tokenizer(text, return_tensors="pt")
input_ids = encodings.input_ids[0]
print(f"  Tokenized: {len(input_ids):,} tokens\n")

# chunk into overlapping sequences of max_length
max_length = CONFIG["max_length"]
sequences  = []
stride     = max_length // 2   # 50% overlap so no context is wasted
for i in range(0, len(input_ids) - max_length, stride):
    sequences.append(input_ids[i : i + max_length])

print(f"  Training sequences: {len(sequences)}")

# =============================================================================
# 3. Training loop
# =============================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

# cosine LR decay
def get_lr(step, total):
    progress = step / total
    return CONFIG["lr"] * (0.1 + 0.9 * (1 - progress))

print(f"\nFine-tuning for {CONFIG['steps']} steps...")
print(f"{'Step':>6}  {'Loss':>8}  {'LR':>10}  {'Time':>8}  ETA")
print("-" * 50)

t0 = time.time()
for step in range(CONFIG["steps"]):
    # update learning rate
    lr = get_lr(step, CONFIG["steps"])
    for g in optimizer.param_groups:
        g["lr"] = lr

    # random batch
    indices = torch.randint(len(sequences), (CONFIG["batch_size"],))
    batch   = torch.stack([sequences[i] for i in indices])   # (B, max_length)

    # in causal LM, the labels are the same as the inputs
    outputs = model(batch, labels=batch)
    loss    = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % CONFIG["print_every"] == 0 or step == CONFIG["steps"] - 1:
        elapsed = time.time() - t0
        sps     = (step + 1) / elapsed if elapsed > 0 else 1
        remaining = (CONFIG["steps"] - step) / sps
        eta = f"{remaining/60:.0f}m" if remaining > 60 else f"{remaining:.0f}s"
        print(f"{step:>6}  {loss.item():>8.4f}  {lr:>10.2e}  {elapsed:>7.1f}s  ETA {eta}")

# =============================================================================
# 4. Save
# =============================================================================
os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\nSaved to {SAVE_PATH}/")
print("Run  python chat.py  to talk to your robot.")
