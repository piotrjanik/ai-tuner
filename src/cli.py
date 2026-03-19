#!/usr/bin/env python3
"""
Fine-tuning pipeline CLI — unified entry point for all pipeline stages.

Usage:
    python src/cli.py --help
    python src/cli.py train --nice 10 --batch-size 1
    python src/cli.py export-gguf
    python src/cli.py push --gguf
"""

import glob as _glob
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml

app = typer.Typer(
    name="llm-forge",
    help="Fine-tuning pipeline: sources → LoRA → GGUF",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

_CONFIG = "config.yaml"
_VENV_PYTHON = Path(".venv/bin/python")


def _py() -> str:
    return str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


def _run(*cmd, nice: int = 0) -> None:
    prefix = ["nice", "-n", str(nice)] if nice > 0 else []
    subprocess.run([*prefix, *[str(c) for c in cmd]], check=True)


def _export_cfg(config: str) -> dict:
    with open(config) as f:
        cfg = yaml.safe_load(f)
    ec = cfg.get("export", {})
    model_name = ec.get("ollama_model_name", "my-model")
    hf_repo    = ec.get("hf_repo", "")
    quant      = ec.get("gguf_quantization", "Q3_K_M")
    return {"model_name": model_name, "hf_repo": hf_repo,
            "quant": quant, "quant_lower": quant.lower()}


def _llama_convert() -> str:
    paths = _glob.glob("/opt/homebrew/Cellar/llama.cpp/*/libexec/convert_hf_to_gguf.py")
    if not paths:
        typer.echo("llama.cpp not found. Install with: brew install llama.cpp", err=True)
        raise typer.Exit(1)
    return sorted(paths)[-1]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def repos(
    config: str = typer.Option(_CONFIG, "-c", help="Config file"),
):
    """Clone source repos defined in config.yaml."""
    _run(_py(), "src/data/prepare_data.py", "--clone-only", "--config", config)


@app.command()
def data(
    stats: bool = typer.Option(False, "--stats", "-s", help="Print stats without writing files"),
    config: str = typer.Option(_CONFIG, "-c", help="Config file"),
):
    """Extract training data from all sources."""
    args = ["--stats-only"] if stats else []
    _run(_py(), "src/data/prepare_data.py", *args, "--config", config)


@app.command()
def train(
    resume:         bool         = typer.Option(False, "--resume", "-r",  help="Continue from existing adapters"),
    nice:           int          = typer.Option(0,     "--nice",   "-n",  help="CPU priority: 0=normal, 10=polite, 19=background"),
    batch_size:     Optional[int]= typer.Option(None,  "--batch-size", "-b", help="Override batch size (reduce to save memory)"),
    max_seq_length: Optional[int]= typer.Option(None,  "--max-seq-length",   help="Override max sequence length"),
    cuda:           bool         = typer.Option(False, "--cuda", "--unsloth", help="Use Unsloth+QLoRA (NVIDIA GPU / Colab) instead of MLX"),
    config:         str          = typer.Option(_CONFIG, "-c", help="Config file"),
):
    """[bold]Fine-tune[/bold] with LoRA. Auto-tunes batch/seq to fit available memory."""
    script = "src/train/train_cuda.py" if cuda else "src/train/train.py"
    cmd = [_py(), script, "--config", config]
    if resume:          cmd.append("--resume")
    if batch_size:      cmd += ["--batch-size", str(batch_size)]
    if max_seq_length:  cmd += ["--max-seq-length", str(max_seq_length)]
    _run(*cmd, nice=nice)


@app.command()
def export(
    config: str = typer.Option(_CONFIG, "-c", help="Config file"),
):
    """Fuse LoRA adapters into [bold]output/merged/[/bold] (float16)."""
    _run(_py(), "src/export/export_model.py", "--config", config)


@app.command()
def export_gguf(
    quant:      Optional[str] = typer.Option(None, "--quant", "-q", help="Quantization (e.g. Q4_K_M). Default: from config.yaml"),
    config:     str           = typer.Option(_CONFIG, "-c", help="Config file"),
):
    """Convert merged model to GGUF for [bold]Ollama[/bold] and [bold]LM Studio[/bold]."""
    ec = _export_cfg(config)
    if quant:
        ec["quant"]       = quant
        ec["quant_lower"] = quant.lower()

    model_name   = ec["model_name"]
    quant_lower  = ec["quant_lower"]
    quant_upper  = ec["quant"]
    gguf_dir     = Path("output/gguf")
    merged_dir   = Path("output/merged")

    if not merged_dir.exists():
        typer.echo("output/merged/ not found. Run: llm-forge export", err=True)
        raise typer.Exit(1)

    gguf_dir.mkdir(parents=True, exist_ok=True)
    f16_path  = gguf_dir / f"{model_name}-f16.gguf"
    out_path  = gguf_dir / f"{model_name}-{quant_lower}.gguf"

    _run(_py(), _llama_convert(), str(merged_dir), "--outfile", str(f16_path), "--outtype", "f16")
    _run("llama-quantize", str(f16_path), str(out_path), quant_upper)
    f16_path.unlink()
    typer.echo(f"\n✓ {out_path}  ({out_path.stat().st_size // 1_073_741_824:.1f} GB)")


@app.command()
def push(
    gguf:   bool = typer.Option(False, "--gguf", "-g", help="Push GGUF file (for LM Studio / Ollama)"),
    config: str  = typer.Option(_CONFIG, "-c", help="Config file"),
):
    """Push to HuggingFace Hub. Adapters by default, [bold]--gguf[/bold] for GGUF."""
    ec = _export_cfg(config)
    if not ec["hf_repo"]:
        typer.echo("Set export.hf_repo in config.yaml first.", err=True)
        raise typer.Exit(1)

    if gguf:
        gguf_path = Path(f"output/gguf/{ec['model_name']}-{ec['quant_lower']}.gguf")
        if not gguf_path.exists():
            typer.echo(f"GGUF not found at {gguf_path}. Run: llm-forge export-gguf", err=True)
            raise typer.Exit(1)
        _run(_py(), "-m", "huggingface_hub.cli.hf", "upload",
             ec["hf_repo"], str(gguf_path), gguf_path.name, "--repo-type", "model")
    else:
        _run(_py(), "src/export/export_model.py", "--push", "--config", config)


@app.command()
def infer(
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Single question (omit for interactive mode)"),
    config:   str           = typer.Option(_CONFIG, "-c", help="Config file"),
):
    """Run inference with the fine-tuned model."""
    cmd = [_py(), "src/inference/inference.py", "--config", config]
    cmd += ["--question", question] if question else ["--interactive"]
    _run(*cmd)


@app.command()
def clean(
    all_:      bool = typer.Option(False, "--all", "-a",    help="Remove data/, output/, repos/ and .venv"),
    repos_dir: bool = typer.Option(False, "--repos",        help="Remove only repos/"),
):
    """Remove generated files ([bold]data/[/bold] and [bold]output/[/bold] by default)."""
    if repos_dir:
        targets = ["repos"]
    elif all_:
        targets = ["data", "output", "repos", ".venv"]
    else:
        targets = ["data", "output"]

    for name in targets:
        p = Path(name)
        if p.exists():
            shutil.rmtree(p)
            typer.echo(f"Removed {name}/")


if __name__ == "__main__":
    app()
