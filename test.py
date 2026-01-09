#!/usr/bin/env python

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("./qwen3-coder-base", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "./qwen3-coder-amber")
tokenizer = AutoTokenizer.from_pretrained("./qwen3-coder-base", trust_remote_code=True)
def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generate_code("Write an Amber script to calculate the sum of two numbers"))
print(generate_code("Convert this Bash to Amber: echo 'Hello'"))
print(generate_code("Convert this Python to Amber: def hello(): print('Hello')"))
