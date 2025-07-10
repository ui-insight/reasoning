###############################################################
#
# 1_inspect_model.py
# 
# Let's look inside the model!
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
#MODEL_NAME= "Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME= "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 1000

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print(model.config)

print("\n\n")

# print the complete model details
for name, module in model.named_modules():
    print(name, '→', module.__class__.__name__)


