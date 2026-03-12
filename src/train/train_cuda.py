#!/usr/bin/env python3
"""
Fine-tune a model using PyTorch + PEFT (QLoRA) on NVIDIA GPUs.

Drop-in replacement for train.py (MLX) — works on Google Colab, RunPod, Lambda, etc.
Reads the same config.yaml and data/{train,val}.jsonl produced by prepare_data.py.

Usage:
    python src/train/train_cuda.py [--config config.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# GPU memory detection & auto-tuning
# ---------------------------------------------------------------------------

# Base model in 4-bit ≈ 20 GB VRAM.  Each batch×seq unit adds roughly 0.5 GB
# on an 80-layer model with LoRA.  These are conservative estimates.
_MODEL_BASE_GB = 20.0


def _estimate_peak_gb(batch: int, seq: int, grad_ckpt: bool = False) -> float:
    """Rough peak VRAM estimate for QLoRA on Qwen3-32B-4bit."""
    per_sample = (seq / 4096) * 2.5  # ~2.5 GB per sample at seq=4096
    activations = batch * per_sample
    optimizer = 1.5  # AdamW states for LoRA params
    peak = _MODEL_BASE_GB + activations + optimizer
    if grad_ckpt:
        peak = _MODEL_BASE_GB + (peak - _MODEL_BASE_GB) * 0.55
    return peak


def _get_gpu_memory_gb() -> tuple[float, str]:
    """Return (total_vram_gb, gpu_name) for the first CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected — this script requires an NVIDIA GPU")
    props = torch.cuda.get_device_properties(0)
    return props.total_mem / (1 << 30), props.name


def auto_tune(cfg_batch: int, cfg_seq: int, cfg_grad_ckpt: bool = False) -> dict:
    """Pick batch_size, max_seq_length, grad_checkpoint that fit in GPU VRAM."""
    total_gb, device = _get_gpu_memory_gb()
    budget_gb = total_gb * 0.90  # tighter margin than unified memory

    candidates = [
        (cfg_batch, cfg_seq, False),
        (cfg_batch, cfg_seq, True),
        (max(cfg_batch // 2, 1), cfg_seq, False),
        (max(cfg_batch // 2, 1), cfg_seq, True),
        (1, cfg_seq, False),
        (1, cfg_seq, True),
        (1, cfg_seq // 2, False),
        (1, cfg_seq // 2, True),
        (1, 1024, True),
    ]

    for batch, seq, ckpt in candidates:
        if seq < 512:
            continue
        est = _estimate_peak_gb(batch, seq, ckpt)
        if est <= budget_gb:
            changes = []
            if batch != cfg_batch:
                changes.append(f"batch_size {cfg_batch}->{batch}")
            if seq != cfg_seq:
                changes.append(f"max_seq_length {cfg_seq}->{seq}")
            if ckpt and not cfg_grad_ckpt:
                changes.append("enabled grad_checkpoint")
            return {
                "batch_size": batch,
                "max_seq_length": seq,
                "grad_checkpoint": ckpt,
                "peak_est_gb": round(est, 1),
                "budget_gb": round(budget_gb, 1),
                "total_gb": round(total_gb, 1),
                "device": device,
                "reason": ", ".join(changes) if changes else "config values fit",
            }

    return {
        "batch_size": 1, "max_seq_length": 1024, "grad_checkpoint": True,
        "peak_est_gb": round(_estimate_peak_gb(1, 1024, True), 1),
        "budget_gb": round(budget_gb, 1), "total_gb": round(total_gb, 1),
        "device": device,
        "reason": "WARNING: even minimum settings may exceed VRAM",
    }


# ---------------------------------------------------------------------------
# Data loading (ShareGPT JSONL → HF Dataset)
# ---------------------------------------------------------------------------

# Map from the Qwen3 chat template.  Qwen3 uses ChatML:
#   <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n...
# The tokenizer.apply_chat_template handles this automatically.

def load_sharegpt_jsonl(path: Path) -> Dataset:
    """Load ShareGPT-format JSONL into an HF Dataset with 'messages' column."""
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    rows = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            messages = [
                {"role": role_map.get(t["from"], t["from"]), "content": t["value"]}
                for t in ex["conversations"]
            ]
            rows.append({"messages": messages})
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# HuggingFace model IDs for CUDA (full-precision, quantized at load time).
# The MLX config uses "mlx-community/Qwen3-32B-4bit" which is MLX-specific.
_MLX_TO_HF = {
    "mlx-community/Qwen3-32B-4bit": "Qwen/Qwen3-32B",
    "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit": "Qwen/Qwen2.5-Coder-7B-Instruct",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="Continue training from existing adapter checkpoint")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--no-auto-tune", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tc = cfg["training"]
    dc = cfg["data"]

    # Resolve HF model ID from the MLX model name
    mlx_model = tc["base_model"]
    hf_model = _MLX_TO_HF.get(mlx_model, mlx_model)
    print(f"Model: {hf_model} (from config: {mlx_model})")

    cfg_batch = args.batch_size or tc.get("per_device_batch_size", 4)
    cfg_seq = args.max_seq_length or tc.get("max_seq_length", 4096)
    cfg_grad_ckpt = tc.get("grad_checkpoint", False)

    # Auto-tune
    user_overrode_both = args.batch_size is not None and args.max_seq_length is not None
    if not args.no_auto_tune and not user_overrode_both:
        tuned = auto_tune(cfg_batch, cfg_seq, cfg_grad_ckpt)
        print(f"GPU: {tuned['total_gb']:.0f} GB ({tuned['device']}), "
              f"budget: {tuned['budget_gb']:.0f} GB, "
              f"est. peak: {tuned['peak_est_gb']:.0f} GB")
        if tuned["reason"] != "config values fit":
            print(f"Auto-tuned: {tuned['reason']}")
        else:
            print("Auto-tune: config values fit within VRAM budget")
        cfg_batch = tuned["batch_size"]
        cfg_seq = tuned["max_seq_length"]
        cfg_grad_ckpt = tuned["grad_checkpoint"]

    # Load data
    data_dir = Path(dc["output_dir"])
    adapter_path = Path(tc["output_dir"]) / "adapters"
    adapter_path.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_ds = load_sharegpt_jsonl(data_dir / "train.jsonl")
    val_ds = load_sharegpt_jsonl(data_dir / "val.jsonl")
    print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    # Quantization config (4-bit NF4 — matches the MLX 4-bit model quality)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model + tokenizer
    print(f"Loading {hf_model} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager",
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config — mirror the MLX settings
    lora_r = tc.get("lora_r", 16)
    lora_alpha = tc.get("lora_scale", 20.0) * lora_r  # mlx scale = alpha/rank
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=int(lora_alpha),
        lora_dropout=tc.get("lora_dropout", 0.0),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Compute max_steps and warmup from iters (mlx-lm "iters" = steps)
    max_steps = tc.get("iters", 1000)
    warmup_steps = tc.get("warmup_steps", 0)

    training_args = TrainingArguments(
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
        gradient_checkpointing=cfg_grad_ckpt,
        gradient_accumulation_steps=max(4 // cfg_batch, 1),  # effective batch ~4
        bf16=True,
        seed=tc.get("seed", 42),
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    # SFTTrainer handles chat template formatting automatically
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        max_seq_length=cfg_seq,
    )

    if args.resume:
        print("Resuming from last checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save final adapter
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapters saved to {adapter_path}")


if __name__ == "__main__":
    main()
