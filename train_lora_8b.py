#!/usr/bin/env python3
"""LoRA training script for 8b models."""

import argparse
import os
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
    import torch
    torch.cuda.set_per_process_memory_fraction(0.7)
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

VALIDATION_PROMPT = "Convert this Bash to Amber: echo 'Hello'"
VALIDATION_LOG_PATH = "validation.log"

try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


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


def generate_validation(model, tokenizer):
    model.eval()
    inputs = tokenizer(VALIDATION_PROMPT, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=False)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


class ValidationCallback:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n=== Validation after epoch {state.epoch} ===")
        generated = generate_validation(self.model, self.tokenizer)
        response = generated[len(VALIDATION_PROMPT):].strip()
        log_entry = f"Epoch {state.epoch}: {response}\n"
        with open(VALIDATION_LOG_PATH, "a") as f:
            f.write(log_entry)
        print(f"Validation output: {response}")
        print("=" * 50 + "\n")


def main():
    print("=" * 60)
    print("LoRA Training Script for Qwen 8B Model")
    print("=" * 60)

    parser = argparse.ArgumentParser(
        description="Train LoRA adapters for 8b language models"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint path to resume from",
    )

    args = parser.parse_args()

    print(f"\n[1/6] Configuration loaded:")
    print(f"    - Epochs: {args.epochs}")
    print(f"    - Batch size: {args.batch_size}")
    print(f"    - Learning rate: {args.lr}")

    device_index = 0
    gpu_total = get_gpu_total_bytes(device_index)
    ram_total = get_system_ram_bytes()

    gpu_gb = human_gb(gpu_total)
    ram_gb = human_gb(ram_total)

    print(f"\n[2/6] Hardware detection:")
    print(f"    - GPU: {gpu_gb:.1f} GB VRAM")
    print(f"    - RAM: {ram_gb:.1f} GB")

    use_offload = False
    use_4bit = False

    if not torch.cuda.is_available():
        use_offload = True
        print("    - Using CPU offload mode (no GPU available)")
    else:
        if gpu_gb < 20:
            if ram_gb >= 24:
                use_offload = True
                print("    - Using CPU offload mode (limited VRAM)")
            else:
                use_4bit = True
                print("    - Using 4-bit quantization (very limited VRAM)")
        else:
            print("    - Full precision mode (sufficient VRAM)")

    model_kwargs = dict(trust_remote_code=True)

    if use_4bit and _HAS_BNB:
        print("\n[3/6] Setting up 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        model_kwargs.update({"quantization_config": bnb_config, "device_map": "auto"})
    elif use_4bit and not _HAS_BNB:
        if ram_gb >= 24:
            use_offload = True

    if use_offload:
        print(f"\n[3/6] Setting up CPU offload...")
        offload_folder = "./offload_8b"
        if os.path.exists(offload_folder):
            shutil.rmtree(offload_folder)
        os.makedirs(offload_folder, exist_ok=True)

        gpu_reserved_gb = max(1, int(gpu_gb * 0.2))
        gpu_allowed_gb = max(0, int(gpu_gb - gpu_reserved_gb))
        ram_allowed_gb = max(1, int(ram_gb * 0.9))
        max_memory = {device_index: f"{gpu_allowed_gb}GB", "cpu": f"{ram_allowed_gb}GB"}
        model_kwargs.update({
            "device_map": "auto",
            "max_memory": max_memory,
            "offload_folder": offload_folder,
        })
    elif not use_4bit:
        if torch.cuda.is_available():
            model_kwargs.update({"device_map": "auto"})
        else:
            model_kwargs.update({"device_map": "cpu"})

    if args.resume_from:
        if not os.path.exists(args.resume_from):
            exit(1)
        print(f"\n[4/6] Resuming from checkpoint: {args.resume_from}")
        model = AutoModelForCausalLM.from_pretrained(args.resume_from, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.resume_from, trust_remote_code=True)
    else:
        print("\n[4/6] Loading base model from ./qwen3-coder-base...")
        base_model_path = "./qwen3-coder-base"
        model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        print("    - Model loaded successfully")

    print("\n[5/6] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("    - LoRA adapter applied")
    model.print_trainable_parameters()

    print("\n[6/6] Loading dataset from ./tokenized_amber_dataset...")
    dataset = load_from_disk("./tokenized_amber_dataset")
    print(f"    - Dataset loaded: {len(dataset)} samples")

    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./qwen3-coder-amber-8b",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_steps=500,
        logging_steps=100,
        fp16=use_4bit,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        callbacks=[ValidationCallback(model, tokenizer)],
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    print("\nSaving LoRA checkpoint...")
    model.save_pretrained("./qwen3-coder-amber-8b")
    tokenizer.save_pretrained("./qwen3-coder-amber-8b")
    print("Training complete. LoRA checkpoint saved to ./qwen3-coder-amber-8b")
    print("=" * 60)


if __name__ == "__main__":
    main()
