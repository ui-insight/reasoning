###############################################################
#
# 0_load_model.py
#
# Simplest script to load and run a transformer LLM 
#
# Luke Sheneman
# University of Idaho, Institute for Interdisciplinary Data Sciences (IIDS)
# sheneman@uidaho.edu
#
# July 2025
#
###############################################################

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose the model name:
MODEL_NAME= "Qwen/Qwen2.5-3B-Instruct"
#MODEL_NAME= "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 1000

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#print(model.dtype)
#print(tokenizer)

# Prepare prompt
prompt = "Write a short poem about Joe Vandal"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#print(inputs)

out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
response = tokenizer.decode(out[0], skip_special_tokens=True)

print(response)

