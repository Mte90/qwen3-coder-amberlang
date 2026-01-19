#!/usr/bin/env python

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

import os
import torch
import shutil

try:
    import pynvml
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

try:
    torch.cuda.set_per_process_memory_fraction(0.7)
except Exception:
    pass


def get_gpu_total_bytes(device_index=0):
    if not torch.cuda.is_available():
        return 0
    if _HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.total
        except Exception:
            pass
    try:
        props = torch.cuda.get_device_properties(device_index)
        return props.total_memory
    except Exception:
        return 0


def get_system_ram_bytes():
    if _HAS_PSUTIL:
        try:
            return psutil.virtual_memory().total
        except Exception:
            pass
    try:
        if hasattr(os, 'sysconf'):
            pages = os.sysconf('SC_PHYS_PAGES')
            pagesize = os.sysconf('SC_PAGE_SIZE')
            return pages * pagesize
    except Exception:
        pass
    return 0


def human_gb(nbytes):
    return nbytes / (1024 ** 3)


device_index = 0
gpu_total = get_gpu_total_bytes(device_index)
ram_total = get_system_ram_bytes()

gpu_gb = human_gb(gpu_total)
ram_gb = human_gb(ram_total)

print(f"Detected GPU total: {gpu_gb:.2f} GB, system RAM: {ram_gb:.2f} GB")

use_offload = False
use_4bit = False

if not torch.cuda.is_available():
    print("CUDA not available: attempting CPU loading")
    use_offload = True
else:
    if gpu_gb < 20:
        if ram_gb >= 24:
            print("GPU VRAM limited and system RAM sufficient: enabling CPU offload")
            use_offload = True
        else:
            print("GPU VRAM limited and system RAM limited: attempting 4-bit loading if available")
            use_4bit = True

try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

model_kwargs = dict(trust_remote_code=True)

if use_4bit and _HAS_BNB:
    print("Loading model in 4-bit mode")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    model_kwargs.update({"quantization_config": bnb_config, "device_map": "auto"})
elif use_4bit and not _HAS_BNB:
    print("bitsandbytes not installed: cannot load in 4-bit")
    if ram_gb >= 24:
        use_offload = True

offload_folder = "./offload"
if use_offload:
    if os.path.exists(offload_folder):
        shutil.rmtree(offload_folder)
    os.makedirs(offload_folder, exist_ok=True)

if use_offload:
    gpu_reserved_gb = max(1, int(gpu_gb * 0.2))
    gpu_allowed_gb = max(0, int(gpu_gb - gpu_reserved_gb))
    ram_allowed_gb = max(1, int(ram_gb * 0.9))
    max_memory = {f"cuda:{device_index}": f"{gpu_allowed_gb}GB", "cpu": f"{ram_allowed_gb}GB"}
    print(f"Loading model with device_map='auto' and max_memory={max_memory}")
    model_kwargs.update({"device_map": "auto", "max_memory": max_memory, "offload_folder": offload_folder})
else:
    if torch.cuda.is_available():
        model_kwargs.update({"device_map": "auto"})
    else:
        model_kwargs.update({"device_map": "cpu"})

print("Loading model")
model = AutoModelForCausalLM.from_pretrained("./qwen3-coder-base", **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained("./qwen3-coder-base", trust_remote_code=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

dataset = load_from_disk("./tokenized_amber_dataset")

training_args = TrainingArguments(
    output_dir="./qwen3-coder-amber",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=5e-5,
    save_steps=500,
    logging_steps=100,
    fp16=True,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
model.save_pretrained("./qwen3-coder-amber")
tokenizer.save_pretrained("./qwen3-coder-amber")
