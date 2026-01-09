#!/usr/bin/env bash

python3 -m venv ./amber-finetune-env
source ./amber-finetune-env/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 # change based on your cuda version
pip install transformers datasets accelerate peft bitsandbytes huggingface_hub

mkdir ./amber-finetune && cd ./amber-finetune

python ./download-qwen-coder.py
