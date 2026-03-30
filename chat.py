"""
chat.py — talk to your robot.

Usage:
    python chat.py
    python chat.py --temp 0.7    (more focused replies)
    python chat.py --temp 1.0    (more creative replies)

Type your message and press Enter. Type 'quit' to exit.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SAVE_PATH = "checkpoints/robot_gpt"


def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No fine-tuned model found at '{path}'.\n"
            "Run  python finetune.py  first."
        )
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.eval()
    return model, tokenizer


def get_reply(model, tokenizer, history, temperature, top_k, device, max_new_tokens=80):
    inputs  = tokenizer(history, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    # trim context if it's getting long (distilgpt2 max = 1024 tokens)
    if input_len > 900:
        trimmed = tokenizer.decode(inputs.input_ids[0, -600:], skip_special_tokens=True)
        inputs  = tokenizer(trimmed, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_new_tokens  = max_new_tokens,
            temperature     = temperature,
            top_k           = top_k,
            do_sample       = True,
            pad_token_id    = tokenizer.eos_token_id,
        )

    new_tokens = output[0, input_len:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # keep only the first line (the bot's turn ends at the next newline)
    reply = reply.split("\n")[0].strip()

    # remove any accidental "User:" continuation
    if "User:" in reply:
        reply = reply.split("User:")[0].strip()

    return reply if reply else "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=SAVE_PATH,  help="Path to fine-tuned model")
    parser.add_argument("--temp",  type=float, default=0.8, help="Temperature")
    parser.add_argument("--topk",  type=int,   default=40,  help="Top-k sampling")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading robot...")
    model, tokenizer = load_model(args.model, device)
    print("Ready.\n")
    print("=" * 50)
    print("  Your robot is online. Say hello!")
    print("  Type 'quit' to exit.")
    print("=" * 50 + "\n")

    history = ""

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bot: goodbye it was nice talking with you!")
            break

        if not user_input:
            continue

        history += f"User: {user_input}\nBot:"
        reply = get_reply(model, tokenizer, history, args.temp, args.topk, device)

        print(f"Bot: {reply}\n")
        history += f" {reply}\n"


if __name__ == "__main__":
    main()
