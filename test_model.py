
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the local model path
model_path = "./sft4_model_merged"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# Define the prompt
prompt = "What is 1 + 1? <metaphor>"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate response
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=4000)

# Decode and print the response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Model Response:", response)

