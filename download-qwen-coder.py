import os
os.environ["HF_HOME"] = "./hf_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model.save_pretrained("./qwen3-coder-base")
tokenizer.save_pretrained("./qwen3-coder-base")
