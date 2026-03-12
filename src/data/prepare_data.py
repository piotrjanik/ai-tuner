#!/usr/bin/env python3
"""
Extract training data from multiple source types for fine-tuning.

Supported sources (configured in config.yaml):
  - go_code: parse Go source files (functions, types, interfaces)
  - spec:    extract sections from Markdown specification files
  - golang_datasets: load public Go datasets (HuggingFace or local JSONL)

Usage:
    python prepare_data.py [--config config.yaml] [--stats-only] [--clone-only]
"""

import os
import re
import json
import random
import argparse
import fnmatch
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

try:
    import yaml
except ImportError:
    print("pyyaml required: pip install pyyaml")
    raise

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it


# ---------------------------------------------------------------------------
# Repo management
# ---------------------------------------------------------------------------

def clone_or_update(url: str, dest: Path, branch: Optional[str] = None) -> bool:
    """Clone repo into repos/ if not already present. Never pushes or writes back upstream."""
    if (dest / ".git").exists():
        print(f"  Already cloned: {dest.name} (skipping network update)")
        return True

    print(f"  Cloning {url} → {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth=1", "--quiet", url, str(dest)]
    if branch:
        cmd[3:3] = ["--branch", branch]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error cloning {url}: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Go Parser
# ---------------------------------------------------------------------------

@dataclass
class GoUnit:
    """A semantic unit extracted from Go source code."""
    kind: str           # func, method, type_struct, type_interface, type_alias
    name: str
    package_name: str
    module_path: str    # Go import path (e.g. github.com/org/repo/pkg)
    file_path: str      # relative to source root
    project_name: str
    doc_comment: str
    code: str           # full declaration including doc comment
    signature: str      # declaration line(s) without body
    body: str           # body only
    receiver_type: str = ""
    line_number: int = 0
    exported: bool = True


class GoParser:
    """Extracts top-level declarations from Go source files."""

    def __init__(self, skip_generated=True, include_tests=True, skip_patterns=None):
        self.skip_generated = skip_generated
        self.include_tests = include_tests
        self.skip_patterns = skip_patterns or []

    # -- public API --

    def parse_directory(self, root_dir: str, project_name: str) -> List[GoUnit]:
        root = Path(root_dir).resolve()
        module_paths = self._find_module_paths(root)

        go_files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "vendor", "node_modules", ".idea", "hack"}
            ]
            for fn in filenames:
                fp = str(os.path.join(dirpath, fn))
                rel = os.path.relpath(fp, str(root))
                if not self._should_process(rel):
                    continue
                go_files.append((fp, rel))

        units: List[GoUnit] = []
        for fp, rel in tqdm(go_files, desc=f"  {project_name}", leave=False):
            if self.skip_generated and self._is_generated(fp):
                continue
            mod = self._resolve_module_path(fp, root, module_paths)
            try:
                units.extend(self._parse_file(fp, project_name, rel, mod))
            except Exception:
                pass  # silently skip unparseable files
        return units

    # -- file-level parsing --

    def _parse_file(self, filepath, project_name, rel_path, module_path):
        with open(filepath, "r", errors="replace") as f:
            content = f.read()
        if not content.strip():
            return []

        lines = content.split("\n")
        # byte offset of each line start
        line_starts = [0]
        for ln in lines[:-1]:
            line_starts.append(line_starts[-1] + len(ln) + 1)

        pkg = ""
        for ln in lines:
            m = re.match(r"\s*package\s+(\w+)", ln)
            if m:
                pkg = m.group(1)
                break

        units = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # skip blank / comment-only lines (comments collected on demand)
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                i += 1
                continue

            # -- func / method --
            if stripped.startswith("func ") and "func()" not in stripped.replace(" ", ""):
                doc = self._collect_doc(lines, i)
                u = self._extract_func(content, lines, line_starts, i,
                                       pkg, project_name, rel_path, module_path, doc)
                if u:
                    units.append(u)
                i = self._line_after_braces(content, line_starts, lines, i)
                continue

            # -- type (not grouped) --
            if stripped.startswith("type ") and not stripped.startswith("type ("):
                doc = self._collect_doc(lines, i)
                tu = self._extract_type(content, lines, line_starts, i,
                                        pkg, project_name, rel_path, module_path, doc)
                if tu:
                    units.append(tu)
                # skip past braces if present
                if self._has_brace_in_range(content, line_starts, lines, i, i + 5):
                    i = self._line_after_braces(content, line_starts, lines, i)
                else:
                    i += 1
                continue

            i += 1

        return units

    # -- declaration extractors --

    def _extract_func(self, content, lines, ls, decl_line,
                      pkg, proj, rel, mod, doc) -> Optional[GoUnit]:
        pos = ls[decl_line]
        brace = self._find_body_brace(content, pos)
        if brace == -1:
            return None
        end = self._find_matching_brace(content, brace)
        if end == -1:
            return None

        sig = content[pos:brace].strip()
        body = content[brace + 1:end].strip()
        code = content[pos:end + 1]

        # receiver?
        rm = re.match(r"func\s+\((\w+)\s+\*?(\w+)(?:\[.*?\])?\)\s+(\w+)", sig)
        if rm:
            recv, name, kind = rm.group(2), rm.group(3), "method"
        else:
            nm = re.match(r"func\s+(\w+)", sig)
            if not nm:
                return None
            recv, name, kind = "", nm.group(1), "func"

        code_with_doc = self._prepend_doc(doc, code)
        return GoUnit(
            kind=kind, name=name, package_name=pkg, module_path=mod,
            file_path=rel, project_name=proj, doc_comment=doc,
            code=code_with_doc, signature=sig, body=body,
            receiver_type=recv, line_number=decl_line + 1,
            exported=name[:1].isupper(),
        )

    def _extract_type(self, content, lines, ls, decl_line,
                      pkg, proj, rel, mod, doc) -> Optional[GoUnit]:
        stripped = lines[decl_line].strip()
        nm = re.match(r"type\s+(\w+)", stripped)
        if not nm:
            return None
        name = nm.group(1)

        if "struct" in stripped:
            kind = "type_struct"
        elif "interface" in stripped:
            kind = "type_interface"
        else:
            kind = "type_alias"

        pos = ls[decl_line]

        # types with body
        if kind != "type_alias":
            brace = self._find_body_brace(content, pos)
            if brace == -1:
                return None
            end = self._find_matching_brace(content, brace)
            if end == -1:
                return None
            code = content[pos:end + 1]
            body = content[brace + 1:end].strip()
            sig = content[pos:brace].strip()
        else:
            # single-line type alias
            nl = content.find("\n", pos)
            code = content[pos:nl].strip() if nl != -1 else content[pos:].strip()
            body = ""
            sig = code

        code_with_doc = self._prepend_doc(doc, code)
        return GoUnit(
            kind=kind, name=name, package_name=pkg, module_path=mod,
            file_path=rel, project_name=proj, doc_comment=doc,
            code=code_with_doc, signature=sig, body=body,
            line_number=decl_line + 1, exported=name[:1].isupper(),
        )

    # -- brace helpers --

    def _find_body_brace(self, content: str, start: int) -> int:
        """Find opening { of a body, skipping struct{}/interface{} in signatures."""
        i, n = start, len(content)
        while i < n:
            c = content[i]
            if c == "/" and i + 1 < n:
                if content[i + 1] == "/":
                    nl = content.find("\n", i)
                    i = nl + 1 if nl != -1 else n
                    continue
                if content[i + 1] == "*":
                    end = content.find("*/", i + 2)
                    i = end + 2 if end != -1 else n
                    continue
            if c == '"':
                i = self._skip_string(content, i)
                continue
            if c == '`':
                j = content.find('`', i + 1)
                i = j + 1 if j != -1 else n
                continue
            if c == '{':
                prefix = content[max(0, i - 12):i].rstrip()
                if prefix.endswith("struct") or prefix.endswith("interface"):
                    close = self._find_matching_brace(content, i)
                    if close != -1:
                        i = close + 1
                        continue
                    return -1
                return i
            i += 1
        return -1

    def _find_matching_brace(self, content: str, pos: int) -> int:
        """Return position of matching } for { at pos."""
        depth, i, n = 0, pos, len(content)
        while i < n:
            c = content[i]
            if c == "/" and i + 1 < n:
                if content[i + 1] == "/":
                    nl = content.find("\n", i)
                    i = nl + 1 if nl != -1 else n
                    continue
                if content[i + 1] == "*":
                    end = content.find("*/", i + 2)
                    i = end + 2 if end != -1 else n
                    continue
            if c == '"':
                i = self._skip_string(content, i)
                continue
            if c == '`':
                j = content.find('`', i + 1)
                i = j + 1 if j != -1 else n
                continue
            if c == "'":
                i += 2 if i + 1 < n and content[i + 1] == '\\' else i + 1
                i = min(i + 1, n)
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1

    @staticmethod
    def _skip_string(content, i):
        """Advance past a double-quoted string starting at i."""
        n = len(content)
        i += 1
        while i < n:
            if content[i] == '\\':
                i += 2
            elif content[i] == '"':
                return i + 1
            else:
                i += 1
        return n

    def _line_after_braces(self, content, ls, lines, start_line):
        """Return the line number after a braced block starting at start_line."""
        pos = ls[start_line]
        brace = content.find("{", pos)
        if brace == -1:
            return start_line + 1
        end = self._find_matching_brace(content, brace)
        if end == -1:
            return start_line + 1
        for k in range(start_line, len(ls)):
            if k + 1 < len(ls) and ls[k] <= end < ls[k + 1]:
                return k + 1
            if k + 1 == len(ls):
                return k + 1
        return len(lines)

    def _has_brace_in_range(self, content, ls, lines, start, end_excl):
        end_excl = min(end_excl, len(lines))
        for k in range(start, end_excl):
            if "{" in lines[k]:
                return True
        return False

    # -- doc comments --

    @staticmethod
    def _collect_doc(lines, decl_line):
        comments = []
        j = decl_line - 1
        while j >= 0:
            s = lines[j].strip()
            if s.startswith("//"):
                comments.insert(0, s[2:].lstrip() if len(s) > 2 else "")
                j -= 1
            elif not s:
                # allow one blank line gap
                if j - 1 >= 0 and lines[j - 1].strip().startswith("//"):
                    j -= 1
                else:
                    break
            else:
                break
        return "\n".join(comments)

    @staticmethod
    def _prepend_doc(doc, code):
        if not doc:
            return code
        doc_block = "\n".join(
            f"// {line}" if line else "//" for line in doc.split("\n")
        )
        return doc_block + "\n" + code

    # -- module path resolution --

    @staticmethod
    def _find_module_paths(root: Path) -> Dict[str, str]:
        result = {}
        for gomod in root.rglob("go.mod"):
            try:
                text = gomod.read_text()
                m = re.search(r"^module\s+(\S+)", text, re.MULTILINE)
                if m:
                    result[str(gomod.parent)] = m.group(1)
            except Exception:
                pass
        return result

    @staticmethod
    def _resolve_module_path(filepath, root, module_paths):
        dirpath = os.path.dirname(os.path.abspath(filepath))
        current = dirpath
        root_s = str(root)
        while current.startswith(root_s):
            if current in module_paths:
                rel = os.path.relpath(dirpath, current)
                if rel == ".":
                    return module_paths[current]
                return module_paths[current] + "/" + rel.replace(os.sep, "/")
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return os.path.relpath(dirpath, root).replace(os.sep, "/")

    # -- filtering --

    def _should_process(self, rel_path):
        if not rel_path.endswith(".go"):
            return False
        if not self.include_tests and rel_path.endswith("_test.go"):
            return False
        for pat in self.skip_patterns:
            if fnmatch.fnmatch(rel_path, pat):
                return False
        return True

    @staticmethod
    def _is_generated(filepath):
        try:
            with open(filepath, "r", errors="replace") as f:
                for i, line in enumerate(f):
                    if i > 5:
                        break
                    low = line.lower()
                    if "code generated" in low or "do not edit" in low or "auto-generated" in low:
                        return True
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Spec Parser (Markdown)
# ---------------------------------------------------------------------------

@dataclass
class SpecUnit:
    """A section extracted from a Markdown specification document."""
    title: str          # heading text
    content: str        # section body (markdown)
    domain: str         # e.g. "OCI Distribution Specification"
    source_name: str    # repo name
    file_path: str      # relative path within the repo
    level: int          # heading level (2 = ##, 3 = ###)


class SpecParser:
    """Extracts sections from Markdown specification files."""

    MIN_CONTENT_CHARS = 100
    MAX_CONTENT_CHARS = 6000

    # Directories to skip when walking spec repos
    SKIP_DIRS = {".git", "vendor", "node_modules", ".github"}
    # File name patterns to skip (case-insensitive)
    SKIP_FILES = {"changelog", "contributing", "license", "codeowners", "authors"}

    def parse_directory(self, root_dir: str, source_name: str, domain: str) -> List[SpecUnit]:
        root = Path(root_dir).resolve()
        units: List[SpecUnit] = []

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS]
            for fn in filenames:
                if not fn.lower().endswith(".md"):
                    continue
                if any(skip in fn.lower() for skip in self.SKIP_FILES):
                    continue
                fp = str(os.path.join(dirpath, fn))
                rel = os.path.relpath(fp, str(root))
                try:
                    units.extend(self._parse_file(fp, source_name, rel, domain))
                except Exception:
                    pass

        return units

    def _parse_file(self, filepath: str, source_name: str, rel_path: str, domain: str) -> List[SpecUnit]:
        with open(filepath, "r", errors="replace") as f:
            content = f.read()

        units: List[SpecUnit] = []
        # Find all h2 (##) and h3 (###) headings
        heading_re = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
        matches = list(heading_re.finditer(content))

        for j, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2).strip()
            # Strip trailing markdown formatting from title (e.g. bold, links)
            title = re.sub(r"[`*\[\]()]", "", title).strip()

            body_start = m.end()
            body_end = matches[j + 1].start() if j + 1 < len(matches) else len(content)
            body = content[body_start:body_end].strip()

            if len(body) < self.MIN_CONTENT_CHARS:
                continue

            if len(body) > self.MAX_CONTENT_CHARS:
                body = body[:self.MAX_CONTENT_CHARS]

            units.append(SpecUnit(
                title=title,
                content=body,
                domain=domain,
                source_name=source_name,
                file_path=rel_path,
                level=level,
            ))

        return units


# ---------------------------------------------------------------------------
# Training Data Generation - System Prompt
# ---------------------------------------------------------------------------

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompt import SYSTEM_PROMPT  # noqa: E402


def _name_to_words(name: str) -> str:
    """CamelCase -> space-separated lowercase words."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return s.lower()


def _conv(user: str, assistant: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": user},
            {"from": "gpt", "value": assistant},
        ]
    }


# ---------------------------------------------------------------------------
# Go Code Training Example Generators
# ---------------------------------------------------------------------------

def gen_code_explanation(u: GoUnit) -> Optional[dict]:
    # Skip undocumented functions — auto-generated answers from CamelCase are low quality
    if not u.doc_comment:
        return None

    if u.kind == "method":
        what = f"the method `{u.name}` on `{u.receiver_type}`"
    else:
        what = f"the function `{u.name}`"

    q = (
        f"Explain what {what} does in package `{u.module_path}` "
        f"({u.project_name}).\n\n```go\n{u.code}\n```"
    )

    parts = [u.doc_comment]
    parts.append(f"\nDefined in `{u.file_path}`, package `{u.package_name}`.")
    parts.append(f"\nSignature: `{u.signature}`")
    return _conv(q, "\n".join(parts))


def gen_code_completion(u: GoUnit) -> Optional[dict]:
    if not u.body or len(u.body) < 30:
        return None

    prompt_lines = []
    if u.doc_comment:
        for line in u.doc_comment.split("\n"):
            prompt_lines.append(f"// {line}" if line else "//")
    prompt_lines.append(u.signature + " {")

    q = "Complete the following Go function body:\n\n```go\n" + "\n".join(prompt_lines) + "\n```"
    a = f"```go\n{u.body}\n```"
    return _conv(q, a)


def gen_type_doc(u: GoUnit) -> Optional[dict]:
    # Skip undocumented types — auto-generated answers from CamelCase are low quality
    if not u.doc_comment:
        return None

    q = (
        f"Describe the `{u.name}` type in `{u.module_path}` "
        f"({u.project_name}).\n\n```go\n{u.code}\n```"
    )

    parts = [u.doc_comment]
    parts.append(f"\nDefined in `{u.file_path}`, package `{u.package_name}`.")

    if u.body:
        members = [l.strip() for l in u.body.split("\n")
                    if l.strip() and not l.strip().startswith("//")]
        label = "field(s)" if u.kind == "type_struct" else "method(s)"
        if members:
            parts.append(f"\nIt has {len(members)} {label}.")

    return _conv(q, "\n".join(parts))


def gen_package_overviews(units: List[GoUnit]) -> List[dict]:
    pkgs: Dict[tuple, List[GoUnit]] = defaultdict(list)
    for u in units:
        pkgs[(u.project_name, u.module_path, u.package_name)].append(u)

    results = []
    for (proj, mod, pkg), members in pkgs.items():
        if len(members) < 3:
            continue

        q = f"Describe the purpose and structure of Go package `{mod}` in the {proj} project."

        types = [u for u in members if u.kind.startswith("type_") and u.exported]
        funcs = [u for u in members if u.kind in ("func", "method") and u.exported]
        files = sorted(set(u.file_path for u in members))

        parts = []
        pkg_doc = next((u.doc_comment for u in members if u.doc_comment and u.kind == "func" and u.name == "init"), "")
        if not pkg_doc:
            pkg_doc = next((u.doc_comment for u in members if u.doc_comment), "")

        if pkg_doc:
            parts.append(pkg_doc[:500])
        else:
            parts.append(f"The `{pkg}` package is part of the {proj} project (`{mod}`).")

        parts.append(f"\nSource files: {', '.join(os.path.basename(f) for f in files[:15])}")
        if types:
            parts.append(f"Exported types ({len(types)}): {', '.join(u.name for u in types[:12])}")
        if funcs:
            parts.append(f"Exported functions/methods ({len(funcs)}): {', '.join(u.name for u in funcs[:12])}")

        results.append(_conv(q, "\n".join(parts)))

    return results


def generate_go_examples(units: List[GoUnit], weights: dict, max_chars: int, min_chars: int) -> List[dict]:
    examples = []

    for u in tqdm(units, desc="  generating Go examples", leave=False):
        if len(u.code) > max_chars or len(u.code) < min_chars:
            continue

        if u.kind in ("func", "method"):
            if weights.get("code_explanation", 1.0) > 0:
                ex = gen_code_explanation(u)
                if ex:
                    examples.append(ex)

            w = weights.get("code_completion", 0.6)
            if w > 0 and random.random() < w:
                ex = gen_code_completion(u)
                if ex:
                    examples.append(ex)

        if u.kind.startswith("type_"):
            w = weights.get("type_documentation", 0.8)
            if w > 0 and random.random() < w:
                ex = gen_type_doc(u)
                if ex:
                    examples.append(ex)

    if weights.get("package_overview", 1.0) > 0:
        examples.extend(gen_package_overviews(units))

    return examples


# ---------------------------------------------------------------------------
# Spec Training Example Generators
# ---------------------------------------------------------------------------

_SPEC_QUESTION_TEMPLATES = [
    "What does the {domain} say about \"{title}\"?",
    "Explain \"{title}\" as described in the {domain}.",
    "According to the {domain}, what is \"{title}\"?",
    "Describe the \"{title}\" concept from the {domain}.",
]


def generate_spec_examples(units: List[SpecUnit], weights: dict) -> List[dict]:
    w = weights.get("spec_explanation", 1.0)
    if w <= 0:
        return []

    examples = []
    for u in units:
        if random.random() > w:
            continue
        tmpl = random.choice(_SPEC_QUESTION_TEMPLATES)
        q = tmpl.format(domain=u.domain, title=u.title)
        examples.append(_conv(q, u.content))

    return examples


# ---------------------------------------------------------------------------
# External Golang Datasets
# ---------------------------------------------------------------------------

def _make_go_example(code: str, doc: str) -> Optional[dict]:
    """Convert a Go function (with optional docstring) to a training example."""
    if not code or len(code) < 40:
        return None
    code = code[:4096]

    if doc and len(doc) > 30:
        q = f"Explain this Go function:\n\n```go\n{code}\n```"
        a = doc[:2000].strip()
        return _conv(q, a)

    # No docstring: generate a code completion example instead
    brace = code.find("{")
    last_brace = code.rfind("}")
    if brace == -1 or last_brace <= brace:
        return None
    sig = code[:brace].strip()
    body = code[brace + 1:last_brace].strip()
    if len(body) < 20:
        return None
    q = f"Complete this Go function body:\n\n```go\n{sig} {{\n```"
    a = f"```go\n{body}\n```"
    return _conv(q, a)


def load_golang_datasets(dataset_cfgs: list) -> List[dict]:
    """Load examples from configured external Golang datasets."""
    all_examples: List[dict] = []

    for ds_cfg in dataset_cfgs:
        ds_type = ds_cfg.get("type", "jsonl")
        name = ds_cfg.get("name", "unknown")
        max_samples = ds_cfg.get("max_samples", 0)

        print(f"\nLoading golang dataset: {name} ({ds_type}) ...")

        if ds_type == "huggingface":
            examples = _load_hf_dataset(ds_cfg)
        elif ds_type == "jsonl":
            examples = _load_jsonl_dataset(ds_cfg)
        else:
            print(f"  Unknown dataset type: {ds_type}, skipping")
            continue

        if max_samples and len(examples) > max_samples:
            random.shuffle(examples)
            examples = examples[:max_samples]

        print(f"  Loaded {len(examples)} examples from {name}")
        all_examples.extend(examples)

    return all_examples


def _load_hf_dataset(cfg: dict) -> List[dict]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("  'datasets' library not installed.")
        print("  Install with: pip install datasets")
        print("  Skipping this HuggingFace dataset.")
        return []

    dataset_id = cfg["dataset_id"]
    subset = cfg.get("subset")
    split = cfg.get("split", "train")
    max_samples = cfg.get("max_samples", 0)
    fmt = cfg.get("format", "code_search_net")
    filter_field = cfg.get("filter_field")
    filter_value = cfg.get("filter_value")

    label = f"{dataset_id}" + (f"/{subset}" if subset else "")
    print(f"  Streaming {label} split={split} format={fmt} ...")
    try:
        ds = load_dataset(dataset_id, subset, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return []

    _role_map = {"user": "human", "assistant": "gpt", "human": "human", "gpt": "gpt", "system": "system"}

    examples: List[dict] = []
    seen = 0
    for item in ds:
        if max_samples and seen >= max_samples:
            break

        # Optional row-level filter (e.g. lang == "go")
        if filter_field and item.get(filter_field) != filter_value:
            continue

        ex: Optional[dict] = None

        if fmt == "messages":
            # Dataset has a `messages` field: list of {role, content}
            raw_msgs = item.get("messages") or []
            if not raw_msgs:
                continue
            convs = [
                {"from": _role_map.get(m["role"], m["role"]), "value": m.get("content", "")}
                for m in raw_msgs
                if m.get("content", "").strip() and m.get("role") != "system"
            ]
            # Always use our system prompt (not the dataset's)
            convs.insert(0, {"from": "system", "value": SYSTEM_PROMPT})
            if len(convs) >= 3:  # system + at least user + assistant
                ex = {"conversations": convs}

        elif fmt == "qa":
            # Dataset has `question` and `answer` (or `input`/`output`) fields
            q = (item.get("question") or item.get("input") or item.get("prompt") or "").strip()
            a = (item.get("answer") or item.get("output") or item.get("response") or "").strip()
            if q and a:
                ex = _conv(q, a)

        elif fmt == "instruction":
            # Dataset has `query`/`answer` or `problem`/`solution` fields
            q = (item.get("query") or item.get("problem") or item.get("instruction") or "").strip()
            a = (item.get("answer") or item.get("solution") or item.get("output") or "").strip()
            if q and a:
                ex = _conv(q, a)

        else:
            # Default: CodeSearchNet-style fields
            code = (item.get("func_code_string") or item.get("whole_func_string")
                    or item.get("code") or item.get("content") or "")
            doc = (item.get("func_documentation_string") or item.get("docstring")
                   or item.get("comment") or "")
            ex = _make_go_example(code, doc)

        if ex:
            examples.append(ex)
            seen += 1

    return examples


def _load_jsonl_dataset(cfg: dict) -> List[dict]:
    path = Path(cfg["path"])
    if not path.exists():
        print(f"  File not found: {path}")
        return []

    examples: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # Already in ShareGPT format
            if "conversations" in item:
                examples.append(item)
                continue

            # CodeSearchNet-like format
            code = (item.get("func_code_string") or item.get("whole_func_string")
                    or item.get("code") or "")
            doc = (item.get("func_documentation_string") or item.get("docstring") or "")
            ex = _make_go_example(code, doc)
            if ex:
                examples.append(ex)

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Prepare training data from Go code, specs, and datasets")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stats-only", action="store_true", help="Print stats, skip writing files")
    ap.add_argument("--clone-only", action="store_true", help="Clone/update repos only, skip parsing")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"Config {cfg_path} not found, using defaults")
        cfg = {"sources": [], "data": {}, "training": {}}

    dc = cfg.get("data", {})
    go_parser = GoParser(
        skip_generated=dc.get("skip_generated", True),
        include_tests=dc.get("include_tests", True),
        skip_patterns=dc.get("skip_patterns", []),
    )
    spec_parser = SpecParser()

    repos_dir = (cfg_path.parent / cfg.get("repos", {}).get("dir", "./repos")).resolve()

    # --- clone / update all remote repos ---
    print("Syncing repos ...")
    for src in cfg.get("sources", []):
        if "url" in src and "path" not in src:
            clone_or_update(src["url"], repos_dir / src["name"], src.get("branch"))

    if args.clone_only:
        print("Done (clone-only mode).")
        return

    # --- parse sources by type ---
    all_go_units: List[GoUnit] = []
    all_spec_units: List[SpecUnit] = []

    for src in cfg.get("sources", []):
        src_type = src.get("type", "go_code")
        name = src["name"]

        if "path" in src:
            abs_path = (cfg_path.parent / src["path"]).resolve()
        else:
            abs_path = repos_dir / name

        if not abs_path.exists():
            print(f"Warning: {abs_path} does not exist, skipping")
            continue

        if src_type == "go_code":
            print(f"\nParsing Go code: {name} ...")
            units = go_parser.parse_directory(str(abs_path), name)
            all_go_units.extend(units)

            funcs = sum(1 for u in units if u.kind == "func")
            methods = sum(1 for u in units if u.kind == "method")
            structs = sum(1 for u in units if u.kind == "type_struct")
            ifaces = sum(1 for u in units if u.kind == "type_interface")
            exported = sum(1 for u in units if u.exported)
            print(f"  {len(units)} units: {funcs} funcs, {methods} methods, "
                  f"{structs} structs, {ifaces} interfaces ({exported} exported)")

        elif src_type == "spec":
            domain = src.get("domain", name)
            print(f"\nParsing spec: {name} ({domain}) ...")
            units = spec_parser.parse_directory(str(abs_path), name, domain)
            all_spec_units.extend(units)
            h2 = sum(1 for u in units if u.level == 2)
            h3 = sum(1 for u in units if u.level == 3)
            print(f"  {len(units)} sections: {h2} h2, {h3} h3")

        else:
            print(f"Warning: unknown source type '{src_type}' for {name}, skipping")

    print(f"\nTotal: {len(all_go_units)} Go units, {len(all_spec_units)} spec sections")

    if args.stats_only:
        return

    # --- generate examples ---
    random.seed(cfg.get("training", {}).get("seed", 42))
    weights = dc.get("weights", {})

    print("\nGenerating training examples ...")
    go_examples = generate_go_examples(
        all_go_units, weights,
        dc.get("max_code_chars", 4096),
        dc.get("min_code_chars", 50),
    )
    print(f"  Go code: {len(go_examples)} examples")

    spec_examples = generate_spec_examples(all_spec_units, weights)
    print(f"  Spec: {len(spec_examples)} examples")

    dataset_examples = load_golang_datasets(cfg.get("golang_datasets", []))
    print(f"  External datasets: {len(dataset_examples)} examples")

    all_examples = go_examples + spec_examples + dataset_examples
    random.shuffle(all_examples)

    # train / val split (5% validation)
    val_n = max(1, int(len(all_examples) * 0.05))
    train_ex = all_examples[val_n:]
    val_ex = all_examples[:val_n]

    out_dir = Path(dc.get("output_dir", "./data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, data in [
        (out_dir / "train.jsonl", train_ex),
        (out_dir / "val.jsonl", val_ex),
    ]:
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(train_ex)} train + {len(val_ex)} val examples to {out_dir}/")

    # save raw Go units for optional inspection / enrichment
    with open(out_dir / "units.jsonl", "w") as f:
        for u in all_go_units:
            f.write(json.dumps({
                "kind": u.kind, "name": u.name,
                "package_name": u.package_name, "module_path": u.module_path,
                "file_path": u.file_path, "project_name": u.project_name,
                "doc_comment": u.doc_comment, "code": u.code,
                "signature": u.signature, "receiver_type": u.receiver_type,
                "exported": u.exported,
            }, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_go_units)} raw Go units to {out_dir}/units.jsonl")


if __name__ == "__main__":
    main()
