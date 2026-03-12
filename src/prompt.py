"""Shared system prompt used across training data generation and inference."""

from pathlib import Path

import yaml

_DEFAULT_PROMPT = (
    "You are an expert coding assistant. You can explain code, interpret "
    "specifications, answer architecture questions, and assist with development."
)

_cached_prompt: str | None = None


def get_system_prompt(config_path: str = "config.yaml") -> str:
    """Load system_prompt from config.yaml, with a sensible default."""
    global _cached_prompt
    if _cached_prompt is not None:
        return _cached_prompt

    p = Path(config_path)
    if p.exists():
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        _cached_prompt = cfg.get("system_prompt", _DEFAULT_PROMPT)
    else:
        _cached_prompt = _DEFAULT_PROMPT
    return _cached_prompt


# For backwards compatibility — modules that do `from prompt import SYSTEM_PROMPT`
SYSTEM_PROMPT = get_system_prompt()
