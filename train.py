#####################################################################################
#
# train.py 
# 
# Fine-tune a quantized LLM (e.g. qwen2.5:3b Instruct) to become a reasoning model 
# via Group Relative Policy Optimization (GRPO).
#
# Luke Sheneman
# Institute for Interdisciplinary Data Sciences (IIDS)
# March, 2025
# sheneman@uidaho.edu
#
# This script leverages unsloth for fune-tuning, using GRPO from the TRL 
#
#   Unsloth:  https://unsloth.ai
#   TLR GRPO Trainer:   https://huggingface.co/docs/trl/main/en/grpo_trainer
#
#####################################################################################

import unsloth
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
import re
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import TrainerCallback, logging
from trl import GRPOConfig, GRPOTrainer

# Set verbosity level for better logging control.
logging.set_verbosity_info()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Patch unsloth for GRPO functionality.
PatchFastRL("GRPO", FastLanguageModel)



# -----------------------------------------------------------------------------
# Hyperparameters and settings.
# -----------------------------------------------------------------------------
MAX_SEQ_LENGTH = 4096
LORA_RANK = 16  # Increased from 8 to 16 for better parameter space
GPU_MEMORY_UTILIZATION = 0.75
LEARNING_RATE = 1e-5  # Lowered learning rate for more stable training
GRPO_GROUP_SIZE = 6  
MAX_PROMPT_LENGTH = 256   # tokens
MAX_COMPLETION_LENGTH = 2048  # tokens



# -----------------------------------------------------------------------------
# Define the system prompt that enforces the desired reasoning format.
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The Assistant first thinks through the reasoning process and then provides the answer. "
    "Ensure that the reasoning is enclosed in <reasoning>...</reasoning> and the final answer in <answer>...</answer>."
)




# -----------------------------------------------------------------------------
# Precompile regular expressions for efficiency.
# -----------------------------------------------------------------------------
FINAL_ANSWER_REGEX = re.compile(r"####\s*(\d+)")
ANSWER_TAG_REGEX = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)




# -----------------------------------------------------------------------------
# Data preprocessing: Convert each dataset example to a conversational format.
# -----------------------------------------------------------------------------
def preprocess_example(example):
    # Extract the final answer after '####'
    answer_match = FINAL_ANSWER_REGEX.search(example["answer"])
    final_answer = answer_match.group(1) if answer_match else example["answer"].strip()
    
    # Clean the question prompt
    clean_question = example["question"].strip()
    
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": clean_question}
        ],
        "ground_truth": final_answer  # Store only the final extracted answer
    }




# -----------------------------------------------------------------------------
# Detailed logging callback for monitoring training progress.
# -----------------------------------------------------------------------------

class DetailedLoggingCallback(TrainerCallback):
    def __init__(self):
        self.running_loss = []
        self.running_rewards = []
        self.reasoning_lengths = []  # Track reasoning output lengths
        self.last_prompt = None
        self.last_completion = None
        self.last_reward = None  # Store last reward value
        self.last_extracted_answer = None
        self.last_ground_truth = None
        self.last_reasoning_tag = False
        self.last_answer_tag = False
        self.last_answer_match = "No match"

    def update_example(self, prompt, completion, reward=None, extracted_answer=None, ground_truth=None):
        """Update the last example details for logging."""
        self.last_prompt = prompt
        self.last_completion = completion
        self.last_reward = reward
        self.last_extracted_answer = extracted_answer
        self.last_ground_truth = ground_truth

        # Use normalize_answer for comparison
        norm_extracted = normalize_answer(extracted_answer) if extracted_answer else ""
        norm_ground_truth = normalize_answer(ground_truth) if ground_truth else ""

        self.last_answer_match = "Exact match" if norm_extracted.strip() == norm_ground_truth.strip() else "Mismatch"

        # Check if the completion includes reasoning and answer tags
        self.last_reasoning_tag = "<reasoning>" in completion and "</reasoning>" in completion
        self.last_answer_tag = "<answer>" in completion and "</answer>" in completion

        # Extract reasoning length
        reasoning_text = re.search(r"<reasoning>(.*?)</reasoning>", completion, re.DOTALL)
        if reasoning_text:
            self.reasoning_lengths.append(len(reasoning_text.group(1).split()))  # Store token length


    def on_step_end(self, args, state, control, **kwargs):
        """Log loss, reward, and example details at the end of each step."""
        if state.log_history and "loss" in state.log_history[-1]:
            loss = state.log_history[-1]["loss"]
            self.running_loss.append(loss)

            print(f"Step {state.global_step}: Loss = {loss:.4f}")

            # Compute and print moving averages every 10 steps
            if state.global_step % 10 == 0 or state.global_step < 5:
                avg_loss = sum(self.running_loss[-10:]) / len(self.running_loss[-10:])
                avg_reward = sum(self.running_rewards[-10:]) / len(self.running_rewards[-10:]) if self.running_rewards else 0
                avg_reasoning_length = sum(self.reasoning_lengths[-10:]) / len(self.reasoning_lengths[-10:]) if self.reasoning_lengths else 0

                print(f"Avg Loss (10 steps) = {avg_loss:.4f}")
                print(f"Average Reward (10 steps): {avg_reward:.4f}")
                print(f"Average Reasoning Length (10 steps): {avg_reasoning_length:.1f} tokens")

            # Log the last example completion with additional details
            if self.last_prompt is not None and self.last_completion is not None:
                print("\n----- Example Prompt-Completion -----")
                print(f"PROMPT: {self.last_prompt}")
                print(f"COMPLETION: {self.last_completion}")
                print(f"REWARD: {self.last_reward}")
                print(f"EXTRACTED ANSWER: {self.last_extracted_answer}")
                print(f"GROUND TRUTH: {self.last_ground_truth}")
                print(f"ANSWER MATCH: {self.last_answer_match}")
                print(f"Includes <reasoning>: {self.last_reasoning_tag}")
                print(f"Includes <answer>: {self.last_answer_tag}")
                print("----- End of Example -----\n")

    def update_rewards(self, rewards):
        """Track reward values for averaging."""
        self.running_rewards.extend(rewards)




# -----------------------------------------------------------------------------
# Helper function to extract the final answer from the answer text.
# -----------------------------------------------------------------------------
def extract_final_answer(answer_text):
    """Extract the final numeric value if present; otherwise, the last meaningful phrase."""
    answer_text = answer_text.strip()
    
    # Try extracting special formatted numbers
    special_formats = re.findall(r"(?:answer|result|total)(?:\s+is)?\s*[:=]?\s*([-+]?\d*\.?\d+)", answer_text, re.IGNORECASE)
    if special_formats:
        return special_formats[-1]
    
    # Try extracting the last numeric value
    numbers = re.findall(r"[-+]?\d*\.?\d+", answer_text)
    if numbers:
        return numbers[-1]

    # Otherwise, extract the last non-empty sentence
    sentences = re.split(r"[.!?]", answer_text)
    for sentence in reversed(sentences):
        sentence = sentence.strip()
        if sentence:
            return sentence

    return answer_text



# -----------------------------------------------------------------------------
# Helper function to normalize numeric answers for consistency and comparison
# -----------------------------------------------------------------------------
def normalize_answer(answer):
    answer = answer.strip()
    
    # Extract all numbers
    numbers = re.findall(r"[-+]?\d*\.?\d+", answer)
    if numbers:
        return numbers[-1].lstrip("0")  # Remove leading zeros for consistency
    
    return answer  # Return text if no number is found



# ----------------------------------------------------------------------------
# The GRPO reward function!   
# -----------------------------------------------------------------------------
def reward_function(completions, ground_truth, prompts=None, **kwargs):
    rewards = []
    total_rewards = []
    detailed_stats = {"format_rewards": [], "correctness_rewards": [], "total": []}
    noise_factor = 0.1  # Small random noise for diversity

    for i, (comp, gt) in enumerate(zip(completions, ground_truth)):
        text = comp[0]["content"]

        # Check for reasoning and answer tags
        has_reasoning = "<reasoning>" in text and "</reasoning>" in text
        has_answer = "<answer>" in text and "</answer>" in text

        # Reward for following the expected format
        format_reward = 1.5 * has_reasoning + 1.5 * has_answer
        detailed_stats["format_rewards"].append(format_reward)

        correctness_reward = 0.0
        predicted_answer = "MISSING"
        gt_str = str(gt).strip()

        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            predicted_answer = match.group(1).strip()
    
        # Normalize both extracted answer and ground truth
        norm_pred = normalize_answer(predicted_answer)
        norm_gt = normalize_answer(gt_str)
    
        if norm_pred == norm_gt:
            correctness_reward = 5.0  # Exact match reward
        elif norm_gt.isnumeric() and norm_pred in norm_gt:
            correctness_reward = 2.0  # Partial match (text contains number)

        detailed_stats["correctness_rewards"].append(correctness_reward)

        # Bonus: Reasoning length-based reward
        if has_reasoning:
            reasoning_text = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
            if reasoning_text:
                length_bonus = min(len(reasoning_text.group(1).split()) / 1000, 1.0) * 3.5  # Max 2.5 bonus
                correctness_reward += length_bonus

        # Final reward calculation
        noise = np.random.normal(0, noise_factor)
        total_reward = format_reward + correctness_reward + noise
        detailed_stats["total"].append(total_reward)
        total_rewards.append(total_reward)
        rewards.append(total_reward)

        # Update the callback with details (for logging)
        if i == 0 and prompts is not None and hasattr(trainer, 'callback'):
            prompt_text = prompts[i][-1]["content"] if isinstance(prompts[i], list) else prompts[i]
            trainer.callback.update_example(
                prompt_text, text, reward=total_reward, extracted_answer=predicted_answer, ground_truth=gt_str
            )

    # Print reward statistics
    avg_format = sum(detailed_stats["format_rewards"]) / len(detailed_stats["format_rewards"])
    avg_correctness = sum(detailed_stats["correctness_rewards"]) / len(detailed_stats["correctness_rewards"])
    avg_total = sum(detailed_stats["total"]) / len(detailed_stats["total"])
    reward_std = np.std(total_rewards)
    print(f"🔍 Reward Stats - Format: {avg_format:.2f}, Correctness: {avg_correctness:.2f}, Total: {avg_total:.2f}, StdDev: {reward_std:.4f}")
    print(f"🔢 Raw rewards: {[round(r, 2) for r in total_rewards[:8]]}{'...' if len(total_rewards) > 8 else ''}")

    # Update rewards in callback
    if hasattr(trainer, 'callback') and hasattr(trainer.callback, 'update_rewards'):
        trainer.callback.update_rewards(rewards)

    return rewards



# -----------------------------------------------------------------------------
# Main training function.
# -----------------------------------------------------------------------------
def main():
    # Set random seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and preprocess the GSM8K dataset.
    dataset = load_dataset("gsm8k", "main", split="train")
    dataset = dataset.map(preprocess_example)
    
    print("First two preprocessed examples:")
    for i in range(2):
        print("*******************")
        print(dataset[i])
    
    # -----------------------------------------------------------------------------
    # Initialize the Qwen2.5 3B Instruct model with 4-bit quantization.
    # -----------------------------------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,      # Disable fast inference for compatibility
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    
    # Ensure the tokenizer pads on the left as required by TRL.
    tokenizer.padding_side = "left"
    
    # Apply QLoRA using unsloth's get_peft_model with improved parameters.
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=2 * LORA_RANK,
        lora_dropout=0.0,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
   

 
    # -----------------------------------------------------------------------------
    # Configure GRPO training parameters with vLLM enabled for faster generation.
    # -----------------------------------------------------------------------------
    training_args = GRPOConfig(
        output_dir="./qwen2.5-3b-grpo",
        logging_steps=1,             # Reduced logging frequency for efficiency
        num_generations=GRPO_GROUP_SIZE,
        fp16=False,
        optim="adamw_torch_fused",
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=LEARNING_RATE,
        use_vllm=True,                # Enable vLLM (ensure you have an available GPU for generation)
        warmup_steps=100,
        weight_decay=0.01,
        max_steps=1000,
        save_steps=200,
        save_total_limit=3,
        temperature=0.5,
        beta=0.0,    # kl_weight
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        dataloader_num_workers=1,     # Increase data loader workers for faster preprocessing
    )
    
    # Create a detailed logging callback instance.
    callback = DetailedLoggingCallback()



    
    # -----------------------------------------------------------------------------
    # Initialize the GRPOTrainer with model, reward function, and training dataset.
    # -----------------------------------------------------------------------------
    global trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer  # Use the tokenizer as the processing class.
    )
    
    # Attach and store the callback for use in the reward function.
    trainer.add_callback(callback)
    trainer.callback = callback
    
    # Start training.
    trainer.train()
    
    # Save the final model after training.
    trainer.save_model("./qwen2.5-3b-grpo-final")
    print("Training completed. Final model saved to ./qwen2.5-3b-grpo-final")


if __name__ == "__main__":
    main()

