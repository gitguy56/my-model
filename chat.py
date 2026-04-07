"""
chat.py — talk to Noodle.

Noodle is powered by TinyLlama-1.1B-Chat directly — no fine-tuning.
It thinks up every reply fresh based on the actual conversation.

Usage:
    python chat.py
    python chat.py --temp 0.7   (more focused)
    python chat.py --temp 1.0   (more creative)

Type 'quit' to exit. Type 'memory' to see what Noodle knows about you.
"""

import os
import json
import argparse
import re
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MEMORY_FILE = "memory/memory.json"
CONV_LOG    = "memory/conversations.log"

DEFAULT_MEMORY = {
    "user_name": "Guy",
    "user_facts": [
        "is building a robot from scratch",
        "likes coding",
        "is casual and direct",
        "prefers short replies",
    ],
    "conversation_count": 0,
    "last_seen": None,
}

# =============================================================================
# Memory
# =============================================================================

def load_memory():
    if os.path.exists(MEMORY_FILE):
        return json.load(open(MEMORY_FILE, encoding="utf-8"))
    return DEFAULT_MEMORY.copy()


def save_memory(memory):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    json.dump(memory, open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)


def extract_facts(user_input, memory):
    """Pick up facts from what Guy says and remember them."""
    text  = user_input.lower()
    patterns = [
        (r"i(?:'m| am) (.+)",                   "is {0}"),
        (r"i like (.+)",                          "likes {0}"),
        (r"i love (.+)",                          "loves {0}"),
        (r"i hate (.+)",                          "hates {0}"),
        (r"i don'?t like (.+)",                   "does not like {0}"),
        (r"my (?:fav(?:ourite|orite)?) .+ is (.+)", "favourite is {0}"),
        (r"i (?:have|got) (.+)",                  "has {0}"),
        (r"i(?:'m| am) (\d+)(?: years old)?",     "is {0} years old"),
    ]
    for pattern, template in patterns:
        m = re.search(pattern, text)
        if m:
            raw = m.group(1).rstrip(".,!?").strip()
            if 1 <= len(raw.split()) <= 7:
                fact = template.format(raw)
                if fact not in memory["user_facts"]:
                    memory["user_facts"].append(fact)
                    if len(memory["user_facts"]) > 25:
                        memory["user_facts"] = memory["user_facts"][-25:]
                    return True
    return False


def build_system_prompt(memory):
    """Build the system prompt Noodle uses to know who it is and who Guy is."""
    facts = ", ".join(memory["user_facts"])
    return (
        "You are Noodle, a witty and casual robot companion. "
        "You are talking to Guy. "
        f"Here is what you know about Guy: {facts}. "
        "Keep replies short, casual, and real. "
        "Do not repeat yourself. "
        "Think up your answer based on what Guy actually said — do not give generic responses. "
        "Never say you are an AI language model. You are Noodle, a robot."
    )


def log_conversation(user_input, reply):
    os.makedirs(os.path.dirname(CONV_LOG), exist_ok=True)
    with open(CONV_LOG, "a", encoding="utf-8") as f:
        f.write(f"Guy: {user_input}\nNoodle: {reply}\n")


# =============================================================================
# Model
# =============================================================================

def load_model(device):
    print("Loading Noodle (TinyLlama-1.1B)...")
    print("  First run downloads ~1.1 GB — loads from cache after that.\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


def build_prompt(system_prompt, chat_history, user_input):
    """
    TinyLlama-Chat uses this format:
      <|system|>...</s>
      <|user|>...</s>
      <|assistant|>
    We include the last few turns of history so Noodle has context.
    """
    prompt = f"<|system|>\n{system_prompt}</s>\n"
    for human, bot in chat_history[-6:]:   # last 6 exchanges = enough context
        prompt += f"<|user|>\n{human}</s>\n<|assistant|>\n{bot}</s>\n"
    prompt += f"<|user|>\n{user_input}</s>\n<|assistant|>\n"
    return prompt


def get_reply(model, tokenizer, system_prompt, chat_history, user_input,
              temperature, top_k, device):
    prompt  = build_prompt(system_prompt, chat_history, user_input)
    inputs  = tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=1024).to(device)
    in_len  = inputs.input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            attention_mask = inputs.attention_mask,
            max_new_tokens = 120,
            temperature    = temperature,
            top_k          = top_k,
            top_p          = 0.9,
            do_sample      = True,
            repetition_penalty = 1.3,
            pad_token_id   = tokenizer.eos_token_id,
        )

    new_tokens = output[0, in_len:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # clean up any template leakage
    for stop in ["</s>", "<|user|>", "<|system|>", "<|assistant|>", "Guy:"]:
        if stop in reply:
            reply = reply.split(stop)[0]

    return reply.strip() or "..."


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--topk", type=int,   default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(device)

    memory = load_memory()
    memory["conversation_count"] += 1
    memory["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_memory(memory)

    system_prompt = build_system_prompt(memory)
    chat_history  = []   # list of (user_msg, noodle_msg) tuples

    print(f"Noodle is online. Hey {memory['user_name']}!")
    print("Type 'quit' to exit  |  'memory' to see what Noodle knows\n")

    while True:
        try:
            user_input = input("Guy: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNoodle: later")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Noodle: later Guy")
            break

        if user_input.lower() == "memory":
            print(f"\n--- Noodle knows ---")
            print(f"  Sessions : {memory['conversation_count']}")
            print(f"  Last seen: {memory['last_seen']}")
            for f in memory["user_facts"]:
                print(f"  - Guy {f}")
            print()
            continue

        # pick up new facts from what Guy says
        if extract_facts(user_input, memory):
            save_memory(memory)
            system_prompt = build_system_prompt(memory)

        reply = get_reply(model, tokenizer, system_prompt, chat_history,
                          user_input, args.temp, args.topk, device)

        print(f"Noodle: {reply}\n")

        chat_history.append((user_input, reply))
        log_conversation(user_input, reply)


if __name__ == "__main__":
    main()
