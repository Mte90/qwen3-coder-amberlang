#!/usr/bin/env python

import json
import os
import io
import zipfile
import sys
import subprocess
import requests
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

if len(sys.argv) < 2:
    print("Usage: python prepare_dataset.py <folder_path1> <folder_path2> ...")
    sys.exit(1)

folder_path = Path(sys.argv[1])
output_file = "amber_dataset.jsonl"

if not os.path.exists(output_file):
    print("Parsing Amber scripts")
    with open(output_file, "w") as f:
        for arg in sys.argv[1:]:
            folder_path = Path(arg)
            if not folder_path.exists() or not folder_path.is_dir():
                print(f"Warning: {folder_path} is not a valid directory, skipping.")
                continue
            for ab_file in folder_path.rglob("*.ab"):
                amber_content = ab_file.read_text().strip()
                try:
                    sh_file = ab_file.with_suffix(".sh")
                    subprocess.run(
                        ["/Amber/target/debug/amber", "build", str(ab_file), str(sh_file)],
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    if sh_file.exists():
                        bash_content = sh_file.read_text().strip()
                        example = {
                            "input": f"Convert this Bash to Amber: {bash_content}",
                            "output": amber_content
                        }
                        f.write(json.dumps(example) + "\n")
                    else:
                        print("Amber script not compiled: " + ab_file)
                except subprocess.CalledProcessError as e:
                    print(f"Error compiling {ab_file}: {e}")

    print("Parsing Amber documentation")
    docs_url = "https://github.com/amber-lang/amber-docs/archive/refs/heads/main.zip"
    response = requests.get(docs_url)
    docs_content = ""
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        md_files = [f for f in z.namelist() if f.startswith("amber-docs-main/docs/nightly-alpha/") and f.endswith(".md")]
        for file in sorted(md_files):
            with z.open(file) as f:
                content = f.read().decode('utf-8')
                docs_content += content + "\n\n"

    if docs_content:
        print("Added Amber documentation")
        docs_example = {
            "input": "Here is the Amber programming language documentation:",
            "output": docs_content.strip()
        }
        with open(output_file, "a") as f:
            f.write(json.dumps(docs_example) + "\n")

dataset = load_dataset("json", data_files=output_file)
tokenizer = AutoTokenizer.from_pretrained("./qwen3-coder-base", trust_remote_code=True)

def tokenize_function(examples):
    inputs = tokenizer(
        examples["input"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    outputs = tokenizer(
        examples["output"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.save_to_disk("./tokenized_amber_dataset")
