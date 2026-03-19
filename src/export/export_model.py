#!/usr/bin/env python3
"""
Export the fine-tuned model for deployment.

Modes:
  (default)  Fuse LoRA adapters into output/merged/ (float16 MLX model)
  --gguf     Export directly to GGUF for Ollama → output/gguf/<name>.gguf
  --push     Push output/merged/ to HuggingFace Hub (requires: huggingface-cli login)

Usage:
    python export_model.py                  # fuse to output/merged/
    python export_model.py --gguf           # fuse + export GGUF
    python export_model.py --push           # push merged model to HF Hub
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _is_peft_adapter(adapter_path: Path) -> bool:
    """Return True if adapters are PEFT format (from Unsloth/transformers)."""
    return (adapter_path / "adapter_config.json").exists()


def _fuse_mlx(base_model: str, adapter_path: Path, merged_path: Path,
              gguf_path: Path | None = None):
    """Fuse MLX LoRA adapters using mlx_lm."""
    if gguf_path:
        cmd = [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model",        base_model,
            "--adapter-path", str(adapter_path),
            "--export-gguf",
            "--gguf-path",    str(gguf_path),
        ]
    else:
        cmd = [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model",        base_model,
            "--adapter-path", str(adapter_path),
            "--save-path",    str(merged_path),
            "--dequantize",   # save as float16 for GGUF compatibility
        ]
    print(" ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)


def _fuse_peft(base_model: str, adapter_path: Path, merged_path: Path):
    """Fuse PEFT LoRA adapters (from Unsloth) using transformers."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="cpu",
    )
    print(f"Loading PEFT adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print("Merging adapters...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_path}...")
    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_path))
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(str(merged_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--gguf", action="store_true", help="Export GGUF for Ollama")
    ap.add_argument("--push", action="store_true", help="Push merged model to HuggingFace Hub")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tc = cfg["training"]
    ec = cfg.get("export", {})

    output_dir = Path(tc["output_dir"])
    adapter_path = output_dir / "adapters"
    merged_path = output_dir / "merged"
    model_name = ec.get("ollama_model_name", "my-model")
    is_peft = _is_peft_adapter(adapter_path)

    if not adapter_path.exists():
        print(f"No adapters found at {adapter_path}. Run train first.")
        sys.exit(1)

    # Resolve which base model was used for training
    if is_peft:
        base_model = tc.get("cuda_model") or tc["base_model"]
        print(f"Detected PEFT adapters (Unsloth backend), base model: {base_model}")
    else:
        base_model = tc["base_model"]
        print(f"Detected MLX adapters, base model: {base_model}")

    if args.gguf:
        gguf_path = output_dir / "gguf" / f"{model_name}.gguf"
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        if is_peft:
            # PEFT → merge first, then convert with llama.cpp
            print(f"Fusing PEFT adapters into {merged_path} ...")
            _fuse_peft(base_model, adapter_path, merged_path)
            print(f"\nMerged model saved to {merged_path}")
            print(f"Convert to GGUF with: task export:gguf")
        else:
            print(f"Exporting GGUF → {gguf_path} ...")
            _fuse_mlx(base_model, adapter_path, merged_path, gguf_path=gguf_path)
            print(f"\nGGUF saved to {gguf_path}")
            _print_ollama_instructions(gguf_path, model_name,
                                       cfg.get("system_prompt", ""))

    elif args.push:
        hf_repo = ec.get("hf_repo", "")
        if not hf_repo:
            print("Set export.hf_repo in config.yaml first (e.g. 'your-username/my-model')")
            sys.exit(1)
        if not adapter_path.exists():
            print(f"No adapters found at {adapter_path}. Run train first.")
            sys.exit(1)
        _write_adapter_readme(adapter_path, base_model, hf_repo, is_peft=is_peft)
        cmd = [
            sys.executable, "-m", "huggingface_hub.cli.hf",
            "upload", hf_repo, str(adapter_path), ".",
            "--repo-type", "model",
        ]
        print(f"Pushing adapters ({adapter_path}) → hf.co/{hf_repo} ...")
        print(" ".join(cmd) + "\n")
        subprocess.run(cmd, check=True)
        print(f"\nAdapters available at: https://huggingface.co/{hf_repo}")
        if is_peft:
            print(f"\nLoad with PEFT:")
            print(f'  from peft import PeftModel')
            print(f'  model = PeftModel.from_pretrained(base_model, "{hf_repo}")')
        else:
            print(f"\nLoad with mlx-lm:")
            print(f'  from mlx_lm import load')
            print(f'  model, tok = load("{base_model}", adapter_path="{hf_repo}")')

    else:
        # Default: fuse adapters into merged model
        print(f"Fusing adapters into {merged_path} ...")
        if is_peft:
            _fuse_peft(base_model, adapter_path, merged_path)
        else:
            _fuse_mlx(base_model, adapter_path, merged_path)
        print(f"\nMerged model saved to {merged_path}")
        print("\nNext steps:")
        print(f"  task export:gguf   # convert to GGUF for Ollama")
        print(f"  task push:hf       # push to HuggingFace Hub")


def _write_adapter_readme(adapter_path: Path, base_model: str, hf_repo: str,
                          is_peft: bool = False):
    readme = adapter_path / "README.md"
    if is_peft:
        readme.write_text(f"""---
base_model: {base_model}
library_name: peft
tags: [peft, lora, fine-tune, unsloth]
---

# {hf_repo.split("/")[-1]}

LoRA adapters fine-tuned from [{base_model}](https://huggingface.co/{base_model}) using Unsloth.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{hf_repo}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```
""")
    else:
        readme.write_text(f"""---
base_model: {base_model}
library_name: mlx
tags: [mlx, lora, fine-tune]
---

# {hf_repo.split("/")[-1]}

LoRA adapters fine-tuned from [{base_model}](https://huggingface.co/{base_model}).

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "{base_model}",
    adapter_path="{hf_repo}",
)
```
""")


def _print_ollama_instructions(gguf_path: Path, model_name: str, system_prompt: str = ""):
    modelfile_path = gguf_path.parent / "Modelfile"
    sys_line = f'SYSTEM "{system_prompt}"\n' if system_prompt else ""
    modelfile_content = (
        f"FROM {gguf_path.resolve()}\n"
        f"{sys_line}"
        f"PARAMETER temperature 0.7\n"
    )
    modelfile_path.write_text(modelfile_content)
    print(f"\nModelfile written to {modelfile_path}")
    print(f"\nImport into Ollama:")
    print(f"  ollama create {model_name} -f {modelfile_path}")
    print(f"  ollama run {model_name}")


if __name__ == "__main__":
    main()
