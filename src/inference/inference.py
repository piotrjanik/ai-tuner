#!/usr/bin/env python3
"""
Interactive inference with the fine-tuned model via mlx-lm.

Usage:
    python inference.py -q "How does package X work?"
    python inference.py -i   # interactive mode
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompt import get_system_prompt  # noqa: E402

import yaml
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("-q", "--question", help="Single question")
    ap.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tc = cfg["training"]
    system_prompt = get_system_prompt(args.config)

    adapter_path = Path(tc["output_dir"]) / "adapters"
    print(f"Loading {tc['base_model']} + adapters from {adapter_path} ...")
    model, tokenizer = load(tc["base_model"], adapter_path=str(adapter_path))
    print("Ready.\n")

    def ask(question: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=2048,
            sampler=make_sampler(temp=0.7, top_p=0.9),
            verbose=False,
        )

    if args.question:
        print(ask(args.question))
    elif args.interactive:
        model_name = cfg.get("export", {}).get("ollama_model_name", "assistant")
        print(f"{model_name} (type 'quit' to exit)\n")
        while True:
            try:
                q = input("> ")
                if q.strip().lower() in ("quit", "exit", "q"):
                    break
                if q.strip():
                    print(f"\n{ask(q)}\n")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        # Demo mode: no hardcoded questions — prompt the user
        print("Usage: pass -q 'question' or -i for interactive mode")


if __name__ == "__main__":
    main()
