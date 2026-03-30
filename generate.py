"""
generate.py — load a trained checkpoint and generate text.

Usage:
    python generate.py
    python generate.py --prompt "ROMEO:" --length 500
    python generate.py --prompt "Hello" --temp 0.7 --topk 20

Options:
    --prompt   Starting text for the model  (default: "\\n")
    --length   Number of new characters to generate  (default: 300)
    --temp     Temperature: higher = more creative, lower = more predictable  (default: 0.8)
    --topk     Only sample from the top-K most likely next tokens  (default: 40)
    --ckpt     Path to checkpoint file  (default: checkpoints/ckpt.pt)
"""

import os
import argparse

import torch

from model import GPT


def load_checkpoint(ckpt_path, vocab_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{ckpt_path}'.\n"
            "Run  python train.py  first to train the model."
        )

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    vocab = torch.load(vocab_path, map_location=device)

    cfg = ckpt["config"]
    char_to_num = vocab["char_to_num"]
    num_to_char = vocab["num_to_char"]
    vocab_size  = len(char_to_num)

    model = GPT(
        vocab_size = vocab_size,
        n_embd     = cfg["n_embd"],
        n_heads    = cfg["n_heads"],
        n_layers   = cfg["n_layers"],
        block_size = cfg["block_size"],
        dropout    = 0.0,   # no dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"  Parameters : {model.num_parameters():,}")
    print(f"  Trained for: {ckpt['step']:,} steps")
    print(f"  Val loss   : {ckpt['val_loss']:.4f}")
    return model, char_to_num, num_to_char


def generate(model, prompt, char_to_num, num_to_char, length, temperature, top_k, device):
    # encode the prompt — skip any characters the model doesn't know
    known = [ch for ch in prompt if ch in char_to_num]
    if not known:
        known = ["\n"]
    idx = torch.tensor([[char_to_num[ch] for ch in known]], dtype=torch.long, device=device)

    # generate
    out = model.generate(idx, max_new_tokens=length, temperature=temperature, top_k=top_k)

    # decode
    return "".join(num_to_char[i] for i in out[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint.")
    parser.add_argument("--prompt", type=str,  default="\n",              help="Starting prompt")
    parser.add_argument("--length", type=int,  default=300,               help="Characters to generate")
    parser.add_argument("--temp",   type=float,default=0.8,               help="Temperature (0.1–2.0)")
    parser.add_argument("--topk",   type=int,  default=40,                help="Top-k sampling")
    parser.add_argument("--ckpt",   type=str,  default="checkpoints/ckpt.pt",  help="Checkpoint path")
    args = parser.parse_args()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_path = os.path.join(os.path.dirname(args.ckpt), "vocab.pt")

    model, char_to_num, num_to_char = load_checkpoint(args.ckpt, vocab_path, device)

    print(f"\nPrompt : {repr(args.prompt)}")
    print(f"Length : {args.length} new characters")
    print(f"Temp   : {args.temp}   Top-k: {args.topk}")
    print("\n" + "=" * 60)

    text = generate(
        model, args.prompt, char_to_num, num_to_char,
        args.length, args.temp, args.topk, device
    )
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
