import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import sys
import logging
import torch
import random
import numpy as np
from transformers import TrainingArguments, TrainerCallback
from datasets import load_dataset
from trl import SFTTrainer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # Base model to fine-tune
DATA_FILE = "synthetic_gsm8k_formatted2.jsonl"  # Local training data file
OUTPUT_DIR = "qwen2.5-3b-sft-unsloth-4bit"  # Directory to save the fine-tuned model
EPOCHS = 5 
LEARNING_RATE = 5e-5
SEED = 42
SEQ_LENGTH = 4096
LORA_RANK  = 64
LORA_ALPHA = LORA_RANK * 2
BATCH_SIZE = 32

# Fixed test prompt to monitor generation during training:
TEST_PROMPT = (
    "Instruction:\nLuke is reading a 1200-page book. He read 12 pages yesterday and triple that amount today. "
    "If he plans to read half of the remaining pages tomorrow, how many pages will that be?\n"
    "Response:"
)

# -----------------------------------------------------------------------------
# Global seed for reproducibility
# -----------------------------------------------------------------------------
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data mapping helper function
# -----------------------------------------------------------------------------
def build_training_prompt(example):
    """
    Build a training prompt from a JSONL example.
    The training data contains:
      - "question": The math problem.
      - "answer": The detailed solution formatted with the intrinsic output tags.
    
    Returns a dictionary with key "text" that constructs the prompt.
    """
    question = example.get("question", "").strip()
    answer = example.get("answer", "").strip()
    
    prompt = (
        f"Instruction:\n{question}\n"
        f"Response:\n{answer}"
    )
    return {"text": prompt}

def tokenize_function(example, tokenizer):
    """
    Tokenize the "text" field of the example.
    Returns the tokenized fields and preserves the original "text" field.
    """
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=SEQ_LENGTH,
        return_attention_mask=True  # Explicitly return attention mask.
    )
    tokenized["text"] = example["text"]
    return tokenized

# -----------------------------------------------------------------------------
# Custom Callback for Generation Monitoring & Format Metric
# -----------------------------------------------------------------------------
class FormatMonitoringCallback(TrainerCallback):
    def __init__(self, model, tokenizer, test_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompt = test_prompt
        self.required_tags = ["<metaphor>", "<reasoning>", "<answer>", "<final_answer>"]

    def on_epoch_end(self, args, state, control, **kwargs):
        inputs = self.tokenizer(
            self.test_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=SEQ_LENGTH,
            return_attention_mask=True  # Ensure attention mask is returned.
        ).to(self.model.device)
        # Pass the attention mask explicitly to generate.
        output_ids = self.model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            max_new_tokens=300
        )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        logger.info("=== Generation Sample at Epoch End ===")
        logger.info(output)
        # Count occurrences of each required tag.
        tag_counts = {tag: output.count(tag) for tag in self.required_tags}
        adherence = sum(1 for count in tag_counts.values() if count == 1) / len(self.required_tags)
        if not output.startswith("<metaphor>"):
            logger.warning("Output does not start with <metaphor>!")
        if not output.endswith("</final_answer>"):
            logger.warning("Output does not end with </final_answer>!")
        logger.info(f"Tag counts: {tag_counts} | Format adherence: {adherence * 100:.1f}%")
        return control

# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------
def main():
    logger.info(f"Loading base model: {BASE_MODEL} with 4-bit quantization")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=SEQ_LENGTH,
            load_in_4bit=True,  # Enable 4-bit quantization (QLoRA)
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        )
    except Exception as e:
        logger.exception(f"Error loading base model: {e}")
        sys.exit(1)
    
    logger.info("Base model loaded successfully in 4-bit mode.")

    # Set a dedicated pad token if necessary to avoid it being the same as the eos token.
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Added dedicated pad token and resized token embeddings.")

    # Apply QLoRA adapter with a rank of LORA_RANK
    logger.info(f"Applying QLoRA adapter with LoRA rank={LORA_RANK}")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )
    except Exception as e:
        logger.exception(f"Error applying QLoRA adapter: {e}")
        sys.exit(1)
    
    # Load the training dataset from the local JSONL file.
    logger.info(f"Loading training dataset from {DATA_FILE}")
    try:
        dataset = load_dataset("json", data_files=DATA_FILE, split="train")
        logger.info(f"Loaded dataset with {len(dataset)} examples.")
    except Exception as e:
        logger.exception(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Build training prompts (creates a new "text" field from "question" and "answer").
    dataset = dataset.map(build_training_prompt)
    
    # Log a few sample prompts for inspection.
    for i in range(min(3, len(dataset))):
        logger.info(f"Sample {i} prompt:\n{dataset[i]['text']}\n{'-'*50}")
    
    # Tokenize the dataset and remove the original fields.
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["question", "answer"]
    )
    
    # Split dataset into training and evaluation sets.
    split_dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        evaluation_strategy="epoch",
        logging_steps=50,
        save_steps=100,
        save_total_limit=25,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        seed=SEED,
    )

    # Initialize the SFT trainer and include our custom callback.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[FormatMonitoringCallback(model, tokenizer, TEST_PROMPT)]
    )

    logger.info("Starting supervised fine-tuning (SFT) with 4-bit QLoRA...")
    try:
        trainer.train()
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        sys.exit(1)
    
    # Save the fine-tuned model and tokenizer.
    try:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"Fine-tuned model and tokenizer saved to {OUTPUT_DIR}")
    except Exception as e:
        logger.exception(f"Error saving the fine-tuned model: {e}")
        sys.exit(1)
    
    logger.info("4-bit QLoRA SFT fine-tuning completed successfully.")
    logger.info("Proceed to the next stage (reinforcement learning) after verifying the model internalizes the correct format.")

if __name__ == "__main__":
    main()

