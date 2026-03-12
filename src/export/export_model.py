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

    if not adapter_path.exists():
        print(f"No adapters found at {adapter_path}. Run train.py first.")
        sys.exit(1)

    if args.gguf:
        gguf_path = output_dir / "gguf" / f"{model_name}.gguf"
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model",        tc["base_model"],
            "--adapter-path", str(adapter_path),
            "--export-gguf",
            "--gguf-path",    str(gguf_path),
        ]
        print(f"Exporting GGUF → {gguf_path} ...")
        print(" ".join(cmd) + "\n")
        subprocess.run(cmd, check=True)
        print(f"\nGGUF saved to {gguf_path}")
        _print_ollama_instructions(gguf_path, model_name,
                                   cfg.get("system_prompt", ""))

    elif args.push:
        hf_repo = ec.get("hf_repo", "")
        if not hf_repo:
            print("Set export.hf_repo in config.yaml first (e.g. 'your-username/my-model')")
            sys.exit(1)
        if not adapter_path.exists():
            print(f"No adapters found at {adapter_path}. Run train.py first.")
            sys.exit(1)
        # Push only the adapters (few hundred MB), not the full merged model (65GB+).
        # Users load them with: load("<base_model>", adapter_path="<hf_repo>")
        _write_adapter_readme(adapter_path, tc["base_model"], hf_repo)
        cmd = [
            sys.executable, "-m", "huggingface_hub.cli.hf",
            "upload", hf_repo, str(adapter_path), ".",
            "--repo-type", "model",
        ]
        print(f"Pushing adapters ({adapter_path}) → hf.co/{hf_repo} ...")
        print(" ".join(cmd) + "\n")
        subprocess.run(cmd, check=True)
        print(f"\nAdapters available at: https://huggingface.co/{hf_repo}")
        print(f"\nLoad with mlx-lm:")
        print(f'  from mlx_lm import load')
        print(f'  model, tok = load("{tc["base_model"]}", adapter_path="{hf_repo}")')

    else:
        # Default: fuse adapters into float16 merged model
        cmd = [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model",        tc["base_model"],
            "--adapter-path", str(adapter_path),
            "--save-path",    str(merged_path),
            "--dequantize",   # save as float16 for GGUF compatibility
        ]
        print(f"Fusing adapters into {merged_path} ...")
        print(" ".join(cmd) + "\n")
        subprocess.run(cmd, check=True)
        print(f"\nMerged model saved to {merged_path}")
        print("\nNext steps:")
        print(f"  task export:gguf   # convert to GGUF for Ollama")
        print(f"  task push:hf       # push to HuggingFace Hub")


def _write_adapter_readme(adapter_path: Path, base_model: str, hf_repo: str):
    readme = adapter_path / "README.md"
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
