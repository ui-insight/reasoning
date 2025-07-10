###############################################################
#
# 3_train_instruct.py
#
# Fine-tune an instruct-tuned causal LM on WikiText as
# instruction-response pairs.
#
# Luke Sheneman
# University of Idaho, IIDS
# July 2025
#
###############################################################

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE  = "wikitext_train.jsonl"  # your JSONL of {"instruction","input","response"}

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# freeze all parameters in the model
for param in model.parameters():
    param.requires_grad = False

# unfreeze the last transformer block
last_n = 1
layers = model.model.layers
for i in range(len(layers) - last_n, len(layers)):
    for p in layers[i].parameters():
        p.requires_grad = True

# also unfreeze language model (LM) head
for p in model.lm_head.parameters():
    p.requires_grad = True


# print some summary info about trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


# load training data
ds = load_dataset(
    "json",
    data_files={"train": DATA_FILE},
    split="train"
)


# build a prompt for each example from our training set
def make_prompt(example):
    # build a single prompt string
    instr = example.get("instruction","Continue the following text:")
    inp   = example.get("input", "").strip()
    resp  = example["response"].strip()

    # Template:
    prompt = (
        f"### Instruction:\n{instr}\n"
        f"### Input:\n{inp}\n"
        f"### Response:\n"
    )
    # Tokenize prompt+response
    tok = tokenizer(
        prompt + resp,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]

    # build labels: mask prompt tokens
    # find where prompt ends in token IDs:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + input_ids[prompt_len:]
    # pad / truncate labels to same length
    labels = labels[:512] + [-100] * max(0, 512 - len(labels))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized = ds.map(
    make_prompt,
    batched=False,
    remove_columns=ds.column_names
)


# Configure training here
training_args = TrainingArguments(
    output_dir="./qwen2.5-instruct-finetune",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    learning_rate=5e-5,
    bf16=True,
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

# train!!
trainer.train()


# inference test - poem time
prompt = "### Instruction:\nWrite a short poem about Joe Vandal\n### Input:\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out    = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(out[0], skip_special_tokens=True))

