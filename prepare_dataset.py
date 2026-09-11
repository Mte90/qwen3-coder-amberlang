#!/usr/bin/env python
"""Amber fine-tuning dataset generator.

Features:
- ingests .ab files from the awesome-amberlang projects (shallow-cloned locally)
- ingests the Amber stdlib/scripts/setup sources (idiomatic, real code)
- deduplicates near-identical files (normalized content hash)
- compile-filters every Amber file with `amber build` before it enters the dataset
- drops pairs that exceed --max-len instead of truncating them mid-code
- varies the Bash->Amber prompt phrasing (deterministic per file)
- splits documentation into one example per section instead of a single giant row
- emits stats so dataset composition is visible

Usage:
    python prepare_dataset.py [options] [extra_folders_with_.ab_files]

With no positional folders, the standard set from the local Amber checkout is used.
"""

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

AWESOME_REPOS = [
    "https://github.com/ArjixWasTaken/amethyst",
    "https://github.com/nixie-dev/nixie",
    "https://github.com/krissh-wtf/amber-httpclient",
    "https://github.com/Mte90/My-Scripts",
    "https://github.com/zlfn/xylitol",
    "https://github.com/UrbanCoffee/amber-projects",
    "https://github.com/rbtylee/AmberResources",
    "https://github.com/MacioSzekTV/TicTacToe-Amber",
]

DEFAULT_AMBER_REPO = Path("/home/mte90/Desktop/kde/Amber")
DEFAULT_SOURCE_DIRS = [
    "src/std",
    "scripts",
    "setup",
    "src/tests/validity",
    "src/tests/stdlib",
    "src/tests/compiling",
    "src/tests/translating",
    "src/tests/optimizing",
    "src/tests/functional",
    "src/tests/runtime",
    "src/tests/testing",
    "src/tests/io",
]
# Excluded on purpose: src/tests/erroring (intentionally invalid code),
# src/tests/warning (code written to trigger warnings).

PROMPT_TEMPLATES = [
    "Convert this Bash script to Amber:\n{bash}",
    "Translate the following Bash into Amber:\n{bash}",
    "Rewrite this Bash script in the Amber programming language:\n{bash}",
    "Here is a Bash script. Produce the equivalent Amber (.ab) code:\n{bash}",
    "Port this Bash code to Amber:\n{bash}",
    "Convert the Bash below to idiomatic Amber:\n{bash}",
    "Bash input, Amber output:\n{bash}",
    "Show this Bash script rewritten in Amber:\n{bash}",
]

DOCS_URL = "https://github.com/amber-lang/amber-docs/archive/refs/heads/main.zip"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="*", help="extra folders containing .ab files")
    p.add_argument("--amber-repo", type=Path, default=DEFAULT_AMBER_REPO,
                   help="local Amber compiler checkout (default: %(default)s)")
    p.add_argument("--amber-bin", default="amber", help="amber compiler binary (default: %(default)s)")
    p.add_argument("--stdlib-dir", type=Path, default=None,
                   help="export AMBER_PATH to this dir if std imports fail to resolve")
    p.add_argument("--out", default="amber_dataset.jsonl", help="output jsonl path")
    p.add_argument("--max-len", type=int, default=2048,
                   help="drop pairs whose token count exceeds this (0 = disable check)")
    p.add_argument("--no-awesome", action="store_true", help="skip awesome-amberlang repos")
    p.add_argument("--awesome-dir", type=Path, default=Path("awesome_projects"),
                   help="cache dir for cloned awesome repos")
    p.add_argument("--no-docs", action="store_true", help="skip amber-docs ingestion")
    p.add_argument("--limit", type=int, default=0, help="process at most N amber files (smoke test)")
    p.add_argument("--no-tokenize", action="store_true",
                   help="skip producing the tokenized dataset on disk")
    return p.parse_args()


def clone_awesome(awesome_dir: Path) -> list[Path]:
    awesome_dir.mkdir(exist_ok=True)
    dirs = []
    for url in AWESOME_REPOS:
        name = url.rstrip("/").split("/")[-1]
        target = awesome_dir / name
        if (target / ".git").exists():
            print(f"  [awesome] reusing {name}")
        else:
            print(f"  [awesome] cloning {name}")
            r = subprocess.run(["git", "clone", "--depth", "1", "--quiet", url, str(target)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  [awesome] FAILED {name}: {r.stderr.strip()[:120]}")
                continue
        dirs.append(target)
    return dirs


def gather_ab_files(folders: list[Path]) -> list[Path]:
    files = []
    for folder in folders:
        if not folder.exists() or not folder.is_dir():
            print(f"  [warn] skipping missing folder: {folder}")
            continue
        files.extend(f.resolve() for f in folder.rglob("*.ab")
                     if "/.git/" not in str(f) and "/target/" not in str(f))
    return sorted(set(files))


def dedupe(files: list[Path]):
    seen = {}
    unique, duplicates = [], 0
    for f in files:
        normalized = "".join(f.read_text(errors="replace").split())
        h = hashlib.sha256(normalized.encode()).hexdigest()
        if h in seen:
            duplicates += 1
        else:
            seen[h] = f
            unique.append(f)
    return unique, duplicates


def compile_amber(amber_bin: str, ab_file: Path, out_sh: Path, stdlib_dir: Path | None, cwd: Path | None = None):
    import os
    env = dict(os.environ)
    if stdlib_dir is not None:
        env["AMBER_PATH"] = str(stdlib_dir)
    try:
        r = subprocess.run(
            [amber_bin, "build", str(ab_file), str(out_sh)],
            capture_output=True, text=True, timeout=60, cwd=str(cwd or ab_file.parent), env=env,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0 or not out_sh.exists():
        return None, (r.stderr or r.stdout).strip().splitlines()[:1][-1] if (r.stderr or r.stdout).strip() else "no output"
    return out_sh.read_text().strip(), None


def prompt_for(path: Path, bash_content: str) -> str:
    idx = int(hashlib.sha256(str(path).encode()).hexdigest(), 16) % len(PROMPT_TEMPLATES)
    return PROMPT_TEMPLATES[idx].format(bash=bash_content)


def load_token_filter(max_len: int):
    if max_len <= 0:
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("./qwen3-coder-base", trust_remote_code=True)
    except Exception as e:
        print(f"  [warn] tokenizer unavailable ({e}); length filtering disabled")
        return None

    def fits(prompt: str, completion: str) -> bool:
        n_prompt = len(tok(prompt, add_special_tokens=True)["input_ids"])
        n_comp = len(tok(completion, add_special_tokens=False)["input_ids"])
        return n_prompt + n_comp + 1 <= max_len

    return fits


def project_root(ab_file: Path, amber_repo: Path) -> Path:
    """Tests import files relative to the repo root, so resolve it for retry."""
    if amber_repo in ab_file.parents:
        return amber_repo
    for parent in ab_file.parents:
        if (parent / ".git").exists():
            return parent
    return ab_file.parent


def split_into_functions(ab_path: Path) -> list[dict]:
    """Split a large .ab file into per-function examples using /// doc comments as prompts.
    Returns [] for files where extraction is not clean (no functions found)."""
    lines = ab_path.read_text(errors="replace").splitlines()
    imports = [ln for ln in lines if ln.startswith("import ")]
    fun_start = re.compile(r"^(pub )?(exported )?fun (\w+)")

    functions = []
    i = 0
    while i < len(lines):
        doc_lines = []
        j = i
        while j < len(lines) and lines[j].strip().startswith("///"):
            doc_lines.append(lines[j].strip().lstrip("/").strip())
            j += 1
        if j < len(lines) and fun_start.match(lines[j]):
            name = fun_start.match(lines[j]).group(3)
            depth = 0
            started = False
            k = j
            while k < len(lines):
                depth += lines[k].count("{") - lines[k].count("}")
                if "{" in lines[k]:
                    started = True
                if started and depth <= 0:
                    break
                k += 1
            body = "\n".join(lines[j:k + 1]).strip()
            if depth == 0 and body:
                doc_text = " ".join(l for l in doc_lines if l)
                if doc_text:
                    prompt = (f"Write the Amber function `{name}`. {doc_text}")
                else:
                    prompt = f"Write the Amber function `{name}` from module 'std/{ab_path.stem}'."
                functions.append({"name": name, "prompt": prompt, "code": body})
            i = k + 1
        else:
            i = j if j > i else i + 1

    examples = []
    for fn in functions:
        code = "\n".join(imports) + "\n\n" + fn["code"] if imports else fn["code"]
        examples.append({"input": fn["prompt"], "output": code,
                         "source": f"fun/{ab_path.name}:{fn['name']}"})
    return examples


def docs_examples(cache_dir: Path):
    cache_dir.mkdir(exist_ok=True)
    zip_path = cache_dir / "amber-docs.zip"
    if not zip_path.exists():
        print("  [docs] downloading amber-docs")
        response = requests.get(DOCS_URL, timeout=120)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    examples = []
    with zipfile.ZipFile(zip_path) as z:
        md_files = sorted(f for f in z.namelist()
                          if f.startswith("amber-docs-main/docs/nightly-alpha/") and f.endswith(".md"))
        for name in md_files:
            content = z.open(name).read().decode("utf-8")
            title = Path(name).stem
            sections = re.split(r"\n(?=## )", content)
            for section in sections:
                heading = next((ln.lstrip("# ").strip() for ln in section.splitlines()
                                if ln.startswith("#")), title)
                text = section.strip()
                if len(text) < 120:
                    continue
                examples.append({
                    "input": f"Explain the Amber language documentation section '{heading}':",
                    "output": text,
                    "source": f"docs/{title}",
                })
    return examples


def main():
    args = parse_args()

    folders = list(args.folders)
    if not folders:
        folders = [args.amber_repo / d for d in DEFAULT_SOURCE_DIRS]
    folders = [Path(f) for f in folders]

    if not args.no_awesome:
        folders.extend(clone_awesome(args.awesome_dir))

    print(f"Collecting .ab files from {len(folders)} folders")
    files = gather_ab_files(folders)
    print(f"  found: {len(files)}")
    if args.limit:
        files = files[:args.limit]
        print(f"  limited to: {len(files)}")

    files, duplicates = dedupe(files)
    print(f"  after dedupe: {len(files)} (removed {duplicates} exact/near duplicates)")

    stdlib_dir = args.stdlib_dir
    fits = load_token_filter(args.max_len)

    stats = {"compile_fail": 0, "too_long": 0, "pairs": 0, "fn_pairs": 0}
    failures = []
    rows = []

    with tempfile.TemporaryDirectory(prefix="amber_ds_") as scratch:
        scratch = Path(scratch)
        for i, ab_file in enumerate(files):
            bash_content, err = compile_amber(args.amber_bin, ab_file, scratch / (ab_file.stem + ".sh"), stdlib_dir)
            if bash_content is None and err and "Could not read file" in err:
                bash_content, err = compile_amber(
                    args.amber_bin, ab_file, scratch / (ab_file.stem + ".sh"), stdlib_dir,
                    cwd=project_root(ab_file, args.amber_repo),
                )
            if bash_content is None:
                stats["compile_fail"] += 1
                failures.append((str(ab_file), str(err)[:100]))
                continue
            prompt = prompt_for(ab_file, bash_content)
            completion = ab_file.read_text().strip()
            if fits and not fits(prompt, completion):
                if "/src/std/" in str(ab_file) or "/scripts/" in str(ab_file) or "/setup/" in str(ab_file):
                    for ex in split_into_functions(ab_file):
                        if fits and not fits(ex["input"], ex["output"]):
                            stats["too_long"] += 1
                            continue
                        rows.append(ex)
                        stats["fn_pairs"] += 1
                else:
                    stats["too_long"] += 1
                continue
            rows.append({"input": prompt, "output": completion, "source": str(ab_file)})
            stats["pairs"] += 1
            if (i + 1) % 50 == 0:
                print(f"  processed {i + 1}/{len(files)}")

    stdlib_files = [f for f in files if "/src/std/" in str(f)]
    for f in stdlib_files:
        module = f.stem
        completion = f.read_text().strip()
        prompt = f"Write the complete Amber source of the standard library module 'std/{module}':"
        if fits and not fits(prompt, completion):
            stats["too_long"] += 1
            continue
        rows.append({"input": prompt, "output": completion, "source": f"stdlib/{module}"})
        stats["pairs"] += 1

    if not args.no_docs:
        for ex in docs_examples(Path("hf_cache") / "amber_docs"):
            if fits and not fits(ex["input"], ex["output"]):
                stats["too_long"] += 1
                continue
            rows.append(ex)

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps({"input": row["input"], "output": row["output"], "source": row["source"]}) + "\n")

    print("\n=== stats ===")
    print(f"  .ab files considered:   {len(files)}")
    print(f"  compile failures:       {stats['compile_fail']}")
    print(f"  dropped (too long):     {stats['too_long']}")
    print(f"  function-level NL pairs: {stats['fn_pairs']}")
    print(f"  dataset rows written:   {len(rows)} -> {args.out}")
    if failures:
        print("\n  compile failures (first 10):")
        for path, err in failures[:10]:
            print(f"    {path}: {err}")

    if args.no_tokenize:
        return

    print("\nTokenizing")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./qwen3-coder-base", trust_remote_code=True)
    dataset = load_dataset("json", data_files=str(args.out))

    def tokenize_function(examples):
        input_ids_list, attention_mask_list, labels_list = [], [], []
        for prompt_text, completion_text in zip(examples["input"], examples["output"]):
            prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
            completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
            completion_ids = completion_ids + [tokenizer.eos_token_id]

            max_len = args.max_len if args.max_len > 0 else 4096
            if len(prompt_ids) + len(completion_ids) > max_len:
                continue

            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            pad_len = max_len - len(input_ids)
            input_ids_list.append(input_ids + [tokenizer.pad_token_id] * pad_len)
            attention_mask_list.append([1] * len(input_ids) + [0] * pad_len)
            labels_list.append(labels + [-100] * pad_len)
        return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized.save_to_disk("./tokenized_amber_dataset")
    print(f"  saved tokenized dataset ({len(tokenized['train'])} rows) -> ./tokenized_amber_dataset")


if __name__ == "__main__":
    main()
