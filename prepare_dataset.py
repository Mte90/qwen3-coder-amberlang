#!/usr/bin/env python

import json
import sys
import subprocess
from pathlib import Path
from datasets import load_dataset

if len(sys.argv) < 2:
    print("Usage: python prepare_dataset.py <folder_path1> <folder_path2> ...")
    sys.exit(1)

folder_path = Path(sys.argv[1])
output_file = "amber_dataset.jsonl"

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
