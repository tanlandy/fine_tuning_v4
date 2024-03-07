import os
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# pip install accelerate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from datasets import load_dataset


model_path = "/opt/models/gemma-7b"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    local_files_only=True,
    device_map="auto",
)
text = "Quote: Imagination is more"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
DATASET_NAME = "/opt/models/datasets/english_quotes"
raw_datasets = load_dataset(DATASET_NAME)
data = raw_datasets.map(lambda samples: tokenizer(samples["quote"]), batched=True)

from peft import LoraConfig

lora_config = LoraConfig(
    r=6,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)
from trl import SFTTrainer


def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}"
    return [text]


trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        num_train_epochs=5,
        logging_steps=1,
        output_dir="outputs3",
        optim="adamw_hf",
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
trainer.train()
text = "Quote: Imagination is"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
