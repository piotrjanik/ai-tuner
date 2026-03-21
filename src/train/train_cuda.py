#!/usr/bin/env python3
"""
Fine-tune a model using Unsloth + QLoRA on NVIDIA GPUs.

Uses Unsloth's FastLanguageModel for 2x faster training and 70% less VRAM
compared to standard transformers. Works on Google Colab, Vertex AI, RunPod, etc.

Reads the same config.yaml and data/{train,val}.jsonl produced by prepare_data.py.

Usage:
    python src/train/train_cuda.py [--config config.yaml]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Disable wandb before importing trl/transformers (broken wandb on Colab crashes imports)
os.environ["WANDB_DISABLED"] = "true"

import torch
import yaml
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only, standardize_sharegpt

# ---------------------------------------------------------------------------
# Data loading (ShareGPT JSONL → HF Dataset)
# ---------------------------------------------------------------------------

def load_sharegpt_jsonl(path: Path) -> Dataset:
    """Load ShareGPT-format JSONL into an HF Dataset with 'conversations' column."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="Continue training from existing adapter checkpoint")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tc = cfg["training"]
    dc = cfg["data"]
    ec = cfg.get("export", {})

    # Resolve model ID — prefer cuda_model, fall back to base_model
    model_id = tc.get("cuda_model") or tc["base_model"]
    print(f"Model: {model_id}")

    cfg_batch = args.batch_size or tc.get("per_device_batch_size", 4)
    cfg_seq = args.max_seq_length or tc.get("max_seq_length", 4096)

    # Load model with Unsloth (handles quantization automatically)
    print(f"Loading {model_id} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=cfg_seq,
        dtype=None,          # auto-detect bf16/fp16
        load_in_4bit=True,
    )

    # LoRA config via Unsloth
    lora_r = tc.get("lora_r", 16)
    lora_alpha = int(tc.get("lora_scale", 20.0) * lora_r)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=tc.get("lora_dropout", 0.0),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM than standard
        random_state=tc.get("seed", 42),
        max_seq_length=cfg_seq,
    )

    # Load data
    data_dir = Path(dc["output_dir"])
    adapter_path = Path(tc["output_dir"]) / "adapters"
    adapter_path.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_ds = load_sharegpt_jsonl(data_dir / "train.jsonl")
    val_ds = load_sharegpt_jsonl(data_dir / "val.jsonl")
    print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    # Standardize ShareGPT format and apply chat template
    train_ds = standardize_sharegpt(train_ds)
    val_ds = standardize_sharegpt(val_ds)

    def formatting_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        ) for convo in convos]
        return {"text": texts}

    train_ds = train_ds.map(formatting_func, batched=True)
    val_ds = val_ds.map(formatting_func, batched=True)

    # Training config
    max_steps = tc.get("iters", 1000)
    warmup_steps = tc.get("warmup_steps", 0)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        max_seq_length=cfg_seq,
        args=SFTConfig(
            output_dir=str(adapter_path),
            max_steps=max_steps,
            per_device_train_batch_size=cfg_batch,
            per_device_eval_batch_size=cfg_batch,
            learning_rate=tc.get("learning_rate", 2e-4),
            lr_scheduler_type="cosine" if tc.get("lr_schedule") == "cosine_decay" else "linear",
            warmup_steps=warmup_steps,
            logging_steps=tc.get("logging_steps", 10),
            eval_steps=tc.get("eval_steps", 200),
            eval_strategy="steps",
            save_steps=tc.get("save_steps", 200),
            save_total_limit=3,
            gradient_accumulation_steps=max(4 // cfg_batch, 1),
            optim="adamw_8bit",
            weight_decay=0.001,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            seed=tc.get("seed", 42),
            report_to="none",
            max_grad_norm=1.0,
            dataloader_pin_memory=False,
        ),
    )

    # Train only on assistant responses (mask instruction/user tokens)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    if args.resume:
        print("Resuming from last checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save adapter
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapters saved to {adapter_path}")

    # Push to HuggingFace if configured
    hf_repo = ec.get("hf_repo", "")
    if hf_repo:
        print(f"\nTo push adapters to HuggingFace:")
        print(f"  python src/cli.py push")
        print(f"\nTo push merged GGUF model:")
        print(f"  python src/cli.py push --gguf")


if __name__ == "__main__":
    main()
