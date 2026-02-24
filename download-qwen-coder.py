import os
os.environ["HF_HOME"] = "./hf_cache"

import subprocess
from pathlib import Path

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
output_dir = "./qwen3-coder-base"

print(f"Downloading {model_name} without loading into memory...")

result = subprocess.run(
    [
        "huggingface-cli", "download",
        model_name,
        "--local-dir", output_dir,
        "--local-dir-use-symlinks", "False",
        "--include", "*.safetensors", "*.json", "*.py", "config.json", "tokenizer_config.json"
    ],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print(f"Model saved to {output_dir}")
    print("Done.")
else:
    print(f"Error: {result.stderr}")
    exit(1)
