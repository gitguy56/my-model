"""
chat.py — talk to Noodle.

Usage:
    python chat.py
    python chat.py --temp 0.7

Type your message and press Enter. Type 'quit' to exit.
Noodle remembers things between sessions.
"""

import os
import json
import argparse
import re
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH  = "checkpoints/robot_gpt"
MEMORY_FILE = "memory/memory.json"
CONV_LOG    = "memory/conversations.log"

# =============================================================================
# Memory — persists between sessions
# =============================================================================

DEFAULT_MEMORY = {
    "user_name": "Guy",
    "noodle_name": "Noodle",
    "user_facts": [
        "is building a robot",
        "likes coding",
        "is casual and direct",
    ],
    "conversation_count": 0,
    "last_seen": None,
}


def load_memory():
    if os.path.exists(MEMORY_FILE):
        return json.load(open(MEMORY_FILE, encoding="utf-8"))
    return DEFAULT_MEMORY.copy()


def save_memory(memory):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    json.dump(memory, open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)


def extract_facts(user_input, memory):
    """Pull simple facts out of what Guy says and add them to memory."""
    text  = user_input.lower()
    added = False

    patterns = [
        (r"i (?:am|m) (.+)",          "is {0}"),
        (r"i like (.+)",               "likes {0}"),
        (r"i love (.+)",               "loves {0}"),
        (r"i (?:hate|don't like) (.+)","does not like {0}"),
        (r"my (?:favourite|favorite) .+ is (.+)", "favourite thing is {0}"),
        (r"i (?:have|got) (.+)",       "has {0}"),
        (r"i (?:work|study) (?:in|on|at) (.+)", "works on {0}"),
    ]

    for pattern, template in patterns:
        m = re.search(pattern, text)
        if m:
            raw   = m.group(1).rstrip(".,!?").strip()
            # skip very short or very long matches
            if 2 < len(raw.split()) < 8:
                fact = template.format(raw)
                if fact not in memory["user_facts"]:
                    memory["user_facts"].append(fact)
                    # keep list from growing too large
                    if len(memory["user_facts"]) > 20:
                        memory["user_facts"] = memory["user_facts"][-20:]
                    added = True
                    break

    return added


def build_context(memory):
    """Build the system context injected before every conversation."""
    name  = memory["noodle_name"]
    user  = memory["user_name"]
    facts = ", ".join(memory["user_facts"]) if memory["user_facts"] else ""
    ctx   = (
        f"{name} is a fun casual robot companion. "
        f"{user} is the user. "
    )
    if facts:
        ctx += f"{user} {facts}. "
    return ctx + "\n"


def log_conversation(user_input, reply, memory):
    os.makedirs(os.path.dirname(CONV_LOG), exist_ok=True)
    with open(CONV_LOG, "a", encoding="utf-8") as f:
        f.write(f"Guy: {user_input}\nNoodle: {reply}\n")


# =============================================================================
# Model
# =============================================================================

def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No fine-tuned model at '{path}'.\n"
            "Run  python finetune.py  first."
        )
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.eval()
    return model, tokenizer


def get_reply(model, tokenizer, context, history, temperature, top_k, device,
              max_new_tokens=80):
    prompt = context + history
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=900).to(device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            attention_mask    = inputs.attention_mask,
            max_new_tokens    = max_new_tokens,
            temperature       = temperature,
            top_k             = top_k,
            do_sample         = True,
            pad_token_id      = tokenizer.eos_token_id,
        )

    new_tokens = output[0, input_len:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # keep only the first line (end of Noodle's turn)
    reply = reply.split("\n")[0].strip()

    # cut off if Guy's turn starts bleeding in
    for tag in ["Guy:", "User:"]:
        if tag in reply:
            reply = reply.split(tag)[0].strip()

    return reply if reply else "..."


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--temp",  type=float, default=0.8)
    parser.add_argument("--topk",  type=int,   default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Noodle...")
    model, tokenizer = load_model(args.model, device)

    memory = load_memory()
    memory["conversation_count"] += 1
    memory["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_memory(memory)

    context = build_context(memory)

    print(f"\nNoodle is online. Hey {memory['user_name']}!")
    print("Type 'quit' to exit  |  'memory' to see what Noodle remembers\n")

    history = ""

    while True:
        try:
            user_input = input(f"{memory['user_name']}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNoodle: later")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Noodle: later Guy come back soon")
            break

        if user_input.lower() == "memory":
            print(f"\n--- Noodle remembers ---")
            print(f"  User      : {memory['user_name']}")
            print(f"  Sessions  : {memory['conversation_count']}")
            print(f"  Last seen : {memory['last_seen']}")
            print(f"  Facts     : {memory['user_facts']}")
            print()
            continue

        # learn from what Guy says
        if extract_facts(user_input, memory):
            save_memory(memory)
            context = build_context(memory)   # rebuild context with new fact

        history += f"{memory['user_name']}: {user_input}\n{memory['noodle_name']}:"
        reply = get_reply(model, tokenizer, context, history,
                          args.temp, args.topk, device)

        print(f"Noodle: {reply}\n")

        history += f" {reply}\n"

        # trim history so it does not grow forever (keep last 20 exchanges)
        lines = history.strip().split("\n")
        if len(lines) > 40:
            history = "\n".join(lines[-40:]) + "\n"

        log_conversation(user_input, reply, memory)


if __name__ == "__main__":
    main()
