import os
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/opt/models/01ai/Yi-34B-Chat-4bits"
tokenizer = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True, use_fast=False
)
model = AutoModelForCausalLM.from_pretrained(
    model_path, local_files_only=True, device_map="auto", torch_dtype=torch.float16
).eval()

# Prompt content: "hi"
Prompt = "How to lose weight and gain muscle?"
messages = [{"role": "user", "content": Prompt}]

input_ids = tokenizer.apply_chat_template(
    conversation=messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
output_ids = model.generate(input_ids.to("cuda"))
response = tokenizer.decode(
    output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
)

# Model response: "Hello! How can I assist you today?"
print(response)
