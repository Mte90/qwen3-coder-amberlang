#!/usr/bin/env python

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from transformers import BitsAndBytesConfig

# import resource
# resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))

import torch
torch.cuda.set_per_process_memory_fraction(0.7)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("./qwen3-coder-base", quantization_config=quantization_config, trust_remote_code=True)
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
