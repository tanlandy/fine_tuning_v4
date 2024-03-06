import os
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_path = "/opt/models/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="auto",
)

# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
#
# tokenizer = AutoTokenizer.from_pretrained(
#     model_path,
#     local_files_only=True,
# )
# model = AutoModelForCausalLM.from_pretrained(
#     model_path, local_files_only=True, quantization_config=quantization_config
# )


input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
