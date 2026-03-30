"""
chat.py — talk to your trained model in a back-and-forth conversation.

Usage:
    python chat.py
    python chat.py --ckpt checkpoints/ckpt.pt
    python chat.py --temp 0.7

Type your message and press Enter. Type 'quit' to exit.
"""

import os
import argparse

import torch

from model import GPT


# How the conversation is formatted in the model's "language".
# The model learns to continue text, so we format chat like a script:
#   User: hello
#   Bot: hi there how can i help
#   User: ...
USER_TAG = "User:"
BOT_TAG  = "Bot:"


def load_checkpoint(ckpt_path, vocab_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at '{ckpt_path}'. Run  python train.py  first."
        )
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab = torch.load(vocab_path, map_location=device, weights_only=False)

    cfg         = ckpt["config"]
    char_to_num = vocab["char_to_num"]
    num_to_char = vocab["num_to_char"]
    vocab_size  = len(char_to_num)

    model = GPT(
        vocab_size = vocab_size,
        n_embd     = cfg["n_embd"],
        n_heads    = cfg["n_heads"],
        n_layers   = cfg["n_layers"],
        block_size = cfg["block_size"],
        dropout    = 0.0,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, char_to_num, num_to_char, cfg["block_size"]


def get_reply(model, context, char_to_num, num_to_char, block_size,
              temperature, top_k, device, max_reply_len=200):
    """
    Feed the conversation context into the model, let it generate until
    it produces a newline (end of the bot's turn) or hits max_reply_len.
    """
    # encode context, crop to block_size
    known  = [ch for ch in context if ch in char_to_num]
    if not known:
        known = ["\n"]
    ids = [char_to_num[ch] for ch in known]
    ids = ids[-block_size:]  # keep only the last block_size tokens

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    reply_ids = []
    with torch.no_grad():
        for _ in range(max_reply_len):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # top-k filter
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            import torch.nn.functional as F
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)

            ch = num_to_char.get(next_tok.item(), "")
            if ch == "\n":
                break
            reply_ids.append(next_tok.item())

    return "".join(num_to_char[i] for i in reply_ids).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  default="checkpoints/ckpt.pt")
    parser.add_argument("--temp",  type=float, default=0.8)
    parser.add_argument("--topk",  type=int,   default=40)
    args = parser.parse_args()

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_path = os.path.join(os.path.dirname(args.ckpt), "vocab.pt")

    print("Loading model...")
    model, char_to_num, num_to_char, block_size = load_checkpoint(
        args.ckpt, vocab_path, device
    )
    print("Model ready.\n")
    print("=" * 50)
    print("  Chat with your robot")
    print("  Type 'quit' to exit")
    print("=" * 50)
    print()

    # keep a rolling conversation history as plain text
    history = ""

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if not user_input:
            continue

        # append user turn, then prompt the model with "Bot:"
        history += f"{USER_TAG} {user_input}\n{BOT_TAG}"

        reply = get_reply(
            model, history, char_to_num, num_to_char, block_size,
            args.temp, args.topk, device
        )

        print(f"Bot: {reply}\n")

        # add bot reply to history so next turn has full context
        history += f" {reply}\n"


if __name__ == "__main__":
    main()
