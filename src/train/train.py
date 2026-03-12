#!/usr/bin/env python3
"""
Fine-tune a Qwen model on Go codebase data using Apple MLX (mlx-lm).

Requires: macOS with Apple Silicon (M1/M2/M3/M4).

Usage:
    python train.py [--config config.yaml]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Memory detection & auto-tuning for Apple Silicon
# ---------------------------------------------------------------------------

# Empirical peak-memory profiles measured on Qwen3-32B-4bit (LoRA rank=32,
# 32 layers).  Each entry is (batch_size, max_seq_length, grad_checkpoint,
# estimated_peak_gb).  The estimator interpolates linearly for batch_size and
# quadratically for seq_length between these anchors.
#
# base ≈ 20 GB (model weights in 4-bit)
_MODEL_BASE_GB = 20.0


def _estimate_peak_gb(batch: int, seq: int, grad_ckpt: bool = False) -> float:
    """Rough estimate of peak Metal memory for Qwen3-32B-4bit LoRA training."""
    # Activations scale ~linearly with batch and ~linearly-to-quadratically
    # with seq_length.  KV-cache is batch × seq × layers × head_dim × 2.
    # We use a simplified model calibrated against the OOM at batch=4/seq=4096
    # (peaked >128 GB) and successful runs at batch=2/seq=4096 (~62 GB).
    activations_gb = batch * (seq / 4096) ** 1.4 * 21.0  # ~21 GB per batch@4096
    kv_cache_gb = batch * seq * 80 * 128 * 2 * 2 / 1e9   # 80 layers, 128 head_dim, fp16
    optimizer_gb = 0.5  # LoRA-only optimizer state is small
    peak = _MODEL_BASE_GB + activations_gb + kv_cache_gb + optimizer_gb
    if grad_ckpt:
        peak = _MODEL_BASE_GB + (peak - _MODEL_BASE_GB) * 0.65  # ~35% savings
    return peak


def _get_memory_gb() -> tuple[float, str]:
    """Return (total_unified_memory_gb, device_name) via MLX."""
    try:
        import mlx.core as mx
        info = mx.device_info()
        return info["memory_size"] / (1 << 30), info.get("device_name", "Apple Silicon")
    except Exception:
        # Fallback: sysctl
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1 << 30), "Apple Silicon"


def auto_tune(cfg_batch: int, cfg_seq: int, cfg_grad_ckpt: bool = False) -> dict:
    """Choose batch_size, max_seq_length, grad_checkpoint that fit in memory.

    Returns a dict with the tuned values and a human-readable reason string.
    """
    total_gb, device = _get_memory_gb()
    budget_gb = total_gb * 0.82  # leave ~18% headroom for OS / other apps

    # Candidates to try, in order of preference (highest quality first).
    candidates = [
        (cfg_batch, cfg_seq,  False),
        (cfg_batch, cfg_seq,  True),              # enable grad checkpoint
        (max(cfg_batch // 2, 1), cfg_seq, False),  # halve batch
        (max(cfg_batch // 2, 1), cfg_seq, True),
        (1, cfg_seq, False),
        (1, cfg_seq, True),
        (1, cfg_seq // 2, False),                 # halve seq length
        (1, cfg_seq // 2, True),
        (1, 1024, True),                          # last resort
    ]

    for batch, seq, ckpt in candidates:
        if seq < 512:
            continue
        est = _estimate_peak_gb(batch, seq, ckpt)
        if est <= budget_gb:
            changes = []
            if batch != cfg_batch:
                changes.append(f"batch_size {cfg_batch}→{batch}")
            if seq != cfg_seq:
                changes.append(f"max_seq_length {cfg_seq}→{seq}")
            if ckpt and not cfg_grad_ckpt:
                changes.append("enabled grad_checkpoint")
            reason = ", ".join(changes) if changes else "config values fit"
            return {
                "batch_size": batch,
                "max_seq_length": seq,
                "grad_checkpoint": ckpt,
                "peak_est_gb": round(est, 1),
                "budget_gb": round(budget_gb, 1),
                "total_gb": round(total_gb, 1),
                "device": device,
                "reason": reason,
            }

    # Nothing fits — return absolute minimum and warn
    return {
        "batch_size": 1,
        "max_seq_length": 1024,
        "grad_checkpoint": True,
        "peak_est_gb": round(_estimate_peak_gb(1, 1024, True), 1),
        "budget_gb": round(budget_gb, 1),
        "total_gb": round(total_gb, 1),
        "device": device,
        "reason": "WARNING: even minimum settings may exceed memory",
    }


def convert_sharegpt_to_mlx(src: Path, dst: Path) -> int:
    """Convert ShareGPT conversations format to mlx-lm messages format."""
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    count = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            ex = json.loads(line)
            messages = [
                {"role": role_map.get(t["from"], t["from"]), "content": t["value"]}
                for t in ex["conversations"]
            ]
            fout.write(json.dumps({"messages": messages}) + "\n")
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="Continue training from existing adapters in output/adapters/")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override per_device_batch_size from config (reduce to save memory)")
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="Override max_seq_length from config (reduce to save memory)")
    ap.add_argument("--no-auto-tune", action="store_true",
                    help="Disable automatic memory-based tuning of batch/seq/grad-checkpoint")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tc = cfg["training"]
    dc = cfg["data"]

    cfg_batch = args.batch_size or tc.get("per_device_batch_size", 4)
    cfg_seq   = args.max_seq_length or tc.get("max_seq_length", 4096)
    cfg_grad_ckpt = tc.get("grad_checkpoint", False)

    # Auto-tune unless both batch & seq were explicitly set on CLI, or --no-auto-tune
    user_overrode_both = args.batch_size is not None and args.max_seq_length is not None
    if not args.no_auto_tune and not user_overrode_both:
        tuned = auto_tune(cfg_batch, cfg_seq, cfg_grad_ckpt)
        print(f"Memory: {tuned['total_gb']:.0f} GB ({tuned['device']}), "
              f"budget: {tuned['budget_gb']:.0f} GB, "
              f"est. peak: {tuned['peak_est_gb']:.0f} GB")
        if tuned["reason"] != "config values fit":
            print(f"Auto-tuned: {tuned['reason']}")
        else:
            print("Auto-tune: config values fit within memory budget")
        cfg_batch     = tuned["batch_size"]
        cfg_seq       = tuned["max_seq_length"]
        cfg_grad_ckpt = tuned["grad_checkpoint"]

    data_dir = Path(dc["output_dir"])
    output_dir = Path(tc["output_dir"])
    mlx_data = output_dir / "mlx_data"
    adapter_path = output_dir / "adapters"
    mlx_data.mkdir(parents=True, exist_ok=True)

    # Convert ShareGPT → mlx-lm messages format
    # mlx-lm expects train.jsonl and valid.jsonl (not val.jsonl)
    for src_name, dst_name in [("train.jsonl", "train.jsonl"), ("val.jsonl", "valid.jsonl")]:
        n = convert_sharegpt_to_mlx(data_dir / src_name, mlx_data / dst_name)
        print(f"Converted {src_name} → {dst_name} ({n} examples)")

    # Write a YAML config for mlx-lm (LoRA rank/scale are config-file-only in 0.30+)
    lora_cfg = {
        "model":             tc["base_model"],
        "train":             True,
        "data":              str(mlx_data),
        "adapter_path":      str(adapter_path),
        "num_layers":        tc.get("lora_num_layers", 16),
        "batch_size":        cfg_batch,
        "iters":             tc.get("iters", 1000),
        "learning_rate":     tc.get("learning_rate", 2e-4),
        "max_seq_length":    cfg_seq,
        "seed":              tc.get("seed", 42),
        "steps_per_report":  tc.get("logging_steps", 10),
        "steps_per_eval":    tc.get("eval_steps", 200),
        "save_every":        tc.get("save_steps", 200),
        "lora_parameters": {
            "rank":    tc.get("lora_r", 16),
            "scale":   tc.get("lora_scale", 20.0),
            "dropout": tc.get("lora_dropout", 0.0),
        },
    }
    if cfg_grad_ckpt:
        lora_cfg["grad_checkpoint"] = True
    warmup = tc.get("warmup_steps", 0)
    if warmup > 0 and tc.get("lr_schedule") == "cosine_decay":
        lora_cfg["lr_schedule"] = {
            "name": "cosine_decay",
            "warmup": warmup,
            "arguments": [tc.get("iters", 1000), tc.get("learning_rate", 1e-5) * 0.1],  # decay_steps, end_lr
        }
    if args.resume and (adapter_path / "adapters.safetensors").exists():
        lora_cfg["resume_adapter_file"] = str(adapter_path / "adapters.safetensors")
        print(f"Resuming from {adapter_path / 'adapters.safetensors'}")
    lora_cfg_path = output_dir / "lora_train_config.yaml"
    with open(lora_cfg_path, "w") as f:
        yaml.dump(lora_cfg, f)

    # mlx-lm 0.30+ uses 'python -m mlx_lm lora' or just 'mlx_lm lora'
    # We use sys.executable to ensure we use the same venv's python
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(lora_cfg_path)]

    total_iters = lora_cfg["iters"]
    print(f"\nStarting training → {adapter_path} ({total_iters} iters)")
    print(" ".join(cmd) + "\n")

    # Let mlx-lm's own tqdm bars (loss calculation etc.) render directly on stderr.
    # We only capture stdout for iter progress lines and overlay our training bar.
    iter_re = re.compile(r"Iter\s+(\d+)")
    loss_re = re.compile(r"Train loss\s+([\d.]+)")
    speed_re = re.compile(r"It/sec\s+([\d.]+)")
    val_re = re.compile(r"Val loss\s+([\d.]+)")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None,
                            bufsize=1, text=True)

    from tqdm import tqdm
    pbar = tqdm(total=total_iters, desc="Training", unit="iter",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                file=sys.stderr)
    last_iter = 0

    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        m = iter_re.search(line)
        if m:
            cur_iter = int(m.group(1))
            pbar.update(cur_iter - last_iter)
            last_iter = cur_iter

            postfix = {}
            lm = loss_re.search(line)
            if lm:
                postfix["loss"] = f"{float(lm.group(1)):.3f}"
            sm = speed_re.search(line)
            if sm:
                postfix["it/s"] = f"{float(sm.group(1)):.2f}"
            vm = val_re.search(line)
            if vm:
                postfix["val"] = f"{float(vm.group(1)):.3f}"
            if postfix:
                pbar.set_postfix(postfix)
        else:
            # Non-iter lines (loading, saving, etc.) — print above the bar
            pbar.clear()
            print(line, flush=True)
            pbar.refresh()

    pbar.close()
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    print(f"\nAdapters saved to {adapter_path}")


if __name__ == "__main__":
    main()
