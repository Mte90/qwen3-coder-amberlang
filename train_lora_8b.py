#!/usr/bin/env python3
"""LoRA training script for 8b models - skeleton with arg parsing."""

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

    device_index = 0
    gpu_total = get_gpu_total_bytes(device_index)
    ram_total = get_system_ram_bytes()

    gpu_gb = human_gb(gpu_total)
    ram_gb = human_gb(ram_total)

    use_offload = False
    use_4bit = False

    if not torch.cuda.is_available():
        use_offload = True
    else:
        if gpu_gb < 20:
            if ram_gb >= 24:
                use_offload = True
            else:
                use_4bit = True

    model_kwargs = dict(trust_remote_code=True)

    if use_4bit and _HAS_BNB:
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

    offload_folder = "./offload_8b"
    if use_offload:
        if os.path.exists(offload_folder):
            shutil.rmtree(offload_folder)
        os.makedirs(offload_folder, exist_ok=True)

    if use_offload:
        gpu_reserved_gb = max(1, int(gpu_gb * 0.2))
        gpu_allowed_gb = max(0, int(gpu_gb - gpu_reserved_gb))
        ram_allowed_gb = max(1, int(ram_gb * 0.9))
        max_memory = {f"cuda:{device_index}": f"{gpu_allowed_gb}GB", "cpu": f"{ram_allowed_gb}GB"}
        model_kwargs.update({"device_map": "auto", "max_memory": max_memory, "offload_folder": offload_folder})
    else:
        if torch.cuda.is_available():
            model_kwargs.update({"device_map": "auto"})
        else:
            model_kwargs.update({"device_map": "cpu"})

    if args.resume_from:
        if not os.path.exists(args.resume_from):
            exit(1)
        print(f"Resuming from {args.resume_from}")
        model = AutoModelForCausalLM.from_pretrained(args.resume_from, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.resume_from, trust_remote_code=True)
    else:
        base_model_path = "./qwen3-coder-base"
        model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    dataset = load_from_disk("./tokenized_amber_dataset")

    training_args = TrainingArguments(
        output_dir="./qwen3-coder-amber-8b",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
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
        callbacks=[ValidationCallback(model, tokenizer)],
    )

    trainer.train()

    model.save_pretrained("./qwen3-coder-amber-8b")
    tokenizer.save_pretrained("./qwen3-coder-amber-8b")

    print("Training complete. LoRA checkpoint saved to ./qwen3-coder-amber-8b")


if __name__ == "__main__":
    main()
