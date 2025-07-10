###############################################################
#
# 2_train.py
#
# Let's freeze parts of the model and fine-tune using Transformers!
#
# Luke Sheneman
# University of Idaho, Institute for Interdisciplinary Data Sciences (IIDS)
# sheneman@uidaho.edu
#
# July 2025
#
###############################################################

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,   # or "auto"
    device_map="auto"
)

# freeze all model parameters by turning off gradient descent flag
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last 3 transformer blocks
last_n = 3
for i in range(len(model.model.layers) - last_n, len(model.model.layers)):
    for param in model.model.layers[i].parameters():
        param.requires_grad = True

# Unfreeze final norm and lm_head
for param in model.model.norm.parameters():
    param.requires_grad = True
for param in model.lm_head.parameters():
    param.requires_grad = True

# Print parameter stats
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# Load and tokenize dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# arguments
training_args = TrainingArguments(
    output_dir="./qwen2.5-last3-finetune",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    learning_rate=5e-5,
    save_steps=50,
    save_total_limit=1,
    bf16=True, 
    logging_strategy="epoch",
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# Prepare prompt
prompt = "Write a short poem about Joe Vandal"

# Tokenize and generate
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#print(inputs)

out = model.generate(**inputs, max_new_tokens=1000)
response = tokenizer.decode(out[0], skip_special_tokens=True)

print(response)
