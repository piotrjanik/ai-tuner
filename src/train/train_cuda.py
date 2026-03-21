#!/usr/bin/env python3
"""
Fine-tune with Unsloth + QLoRA on NVIDIA GPUs (H100, A100, L4, etc).

Reads config.yaml and data/{train,val}.jsonl produced by prepare_data.py.

Usage:
    python src/train/train_cuda.py [--config config.yaml] [--resume]
"""

import argparse
import json
import os
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"

import torch
import yaml
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only

# Unsloth patches SFTTrainer — import after unsloth
from trl import SFTTrainer


def load_sharegpt_jsonl(path: Path) -> Dataset:
    """Load ShareGPT-format JSONL into an HF Dataset."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tc = cfg["training"]
    dc = cfg["data"]

    model_id = tc.get("cuda_model") or tc["base_model"]
    batch_size = args.batch_size or tc.get("per_device_batch_size", 4)
    max_seq_len = args.max_seq_length or tc.get("max_seq_length", 4096)
    lora_r = tc.get("lora_r", 32)
    lora_alpha = int(tc.get("lora_scale", 6.0) * lora_r)
    seed = tc.get("seed", 42)

    data_dir = Path(dc["output_dir"])
    adapter_path = Path(tc["output_dir"]) / "adapters"
    adapter_path.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading {model_id} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_len,
        dtype=None,          # auto bf16 on H100
        load_in_4bit=True,
    )

    # ── LoRA adapters ───────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=tc.get("lora_dropout", 0.05),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=max_seq_len,
    )

    # ── Load and format data ────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = load_sharegpt_jsonl(data_dir / "train.jsonl")
    val_ds = load_sharegpt_jsonl(data_dir / "val.jsonl")
    print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    train_ds = standardize_sharegpt(train_ds)
    val_ds = standardize_sharegpt(val_ds)

    def format_chat(examples):
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in examples["conversations"]
        ]
        return {"text": texts}

    train_ds = train_ds.map(format_chat, batched=True)
    val_ds = val_ds.map(format_chat, batched=True)

    # ── Trainer ─────────────────────────────────────────────────────────────
    max_steps = tc.get("iters", 10000)

    # Auto-tune: halve batch size until it fits, compensate with grad accumulation
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    effective_batch = batch_size
    # 32B 4-bit model needs ~20GB base + ~15GB per batch item at seq_len=4096
    # On 80GB H100 with 32B: batch=2 is safe, batch=4 OOMs
    if gpu_mem_gb < 160:  # single GPU (not multi-GPU with >160GB total)
        max_batch_for_mem = max(1, int((gpu_mem_gb - 25) / 14))
        if batch_size > max_batch_for_mem:
            batch_size = max(1, max_batch_for_mem)
            print(f"Auto-tuned batch_size to {batch_size} (GPU: {gpu_mem_gb:.0f} GB)")
    grad_accum = max(effective_batch // batch_size, 1)

    training_args = TrainingArguments(
        output_dir=str(adapter_path),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=tc.get("learning_rate", 3e-5),
        lr_scheduler_type="cosine" if tc.get("lr_schedule") == "cosine_decay" else "linear",
        warmup_steps=tc.get("warmup_steps", 500),
        weight_decay=0.001,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=tc.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=tc.get("eval_steps", 500),
        save_steps=tc.get("save_steps", 500),
        save_total_limit=3,
        seed=seed,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=training_args,
    )

    # Mask system/user tokens — only train on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # ── Train ───────────────────────────────────────────────────────────────
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name} ({gpu.total_memory / 1e9:.0f} GB)")
    print(f"Config: batch={batch_size}, grad_accum={grad_accum}, seq_len={max_seq_len}, "
          f"lora_r={lora_r}, steps={max_steps}")

    if args.resume:
        print("Resuming from last checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # ── Save ────────────────────────────────────────────────────────────────
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapters saved to {adapter_path}")


if __name__ == "__main__":
    main()