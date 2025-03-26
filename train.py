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
import math
import json
import argparse
from datasets import load_dataset
from transformers import TrainerCallback, logging, default_data_collator
from trl import GRPOConfig, GRPOTrainer

# Set verbosity level for better logging control.
logging.set_verbosity_info()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Patch unsloth for GRPO functionality.
PatchFastRL("GRPO", FastLanguageModel)

# -----------------------------------------------------------------------------
# Hyperparameters and settings.
# -----------------------------------------------------------------------------
MAX_SEQ_LENGTH = 8192
LORA_RANK  = 16 
LORA_ALPHA = 16 
GPU_MEMORY_UTILIZATION = 0.80
LEARNING_RATE = 1e-5  # Lowered learning rate for more stable training
GRPO_GROUP_SIZE = 6  
MAX_PROMPT_LENGTH = 1024   # tokens
MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH  # tokens
DEFAULT_TRAINING_STEPS = 1000  # default additional training steps
TEMPERATURE = 0.7
NUM_SAVE_CHECKPOINTS = 10
SAVE_STEPS = 50
WARMUP_STEPS = 100  # for our 100-step curriculum learning steps

# -----------------------------------------------------------------------------
# Define the system prompt that enforces the desired reasoning format.
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The response must begin with <metaphor>"

# -----------------------------------------------------------------------------
# Precompile regular expressions for efficiency.
# -----------------------------------------------------------------------------
FINAL_ANSWER_REGEX = re.compile(r"####\s*([\d,]+)")
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
        self.last_metaphor_tag = False
        self.last_answer_tag = False
        self.last_final_answer_tag = False
        self.last_extra_text = False
        self.last_answer_match = "No match"
        self.last_completion_length = 0  # Overall completion length (token count)

    def update_example(self, prompt, completion, reward=None, extracted_answer=None, ground_truth=None,
                       includes_reasoning=False, includes_metaphor=False, includes_answer=False, includes_final_answer=False,
                       includes_extra_text=False, completion_length=0):
        self.last_prompt = prompt
        self.last_completion = completion
        self.last_reward = reward
        self.last_extracted_answer = extracted_answer
        self.last_ground_truth = ground_truth
        self.last_completion_length = completion_length

        # Use normalize_answer for comparison
        norm_extracted = normalize_answer(extracted_answer) if extracted_answer else ""
        norm_ground_truth = normalize_answer(ground_truth) if ground_truth else ""
        self.last_answer_match = "Exact match" if norm_extracted.strip() == norm_ground_truth.strip() else "Mismatch"

        # Update tag flags
        self.last_reasoning_tag = includes_reasoning
        self.last_metaphor_tag = includes_metaphor
        self.last_answer_tag = includes_answer
        self.last_final_answer_tag = includes_final_answer
        self.last_extra_text = includes_extra_text

        # ANSI escape codes for colors
        GREEN = "\033[32m"
        RED = "\033[31m"
        RESET = "\033[0m"

        checkmark = f"{GREEN}✔{RESET}"
        cross = f"{RED}✖{RESET}"

        answer_match_status = checkmark if self.last_answer_match == "Exact match" else cross
        reasoning_status = checkmark if self.last_reasoning_tag else cross
        metaphor_status = checkmark if self.last_metaphor_tag else cross
        answer_status = checkmark if self.last_answer_tag else cross
        final_answer_status = checkmark if self.last_final_answer_tag else cross
        extra_text_status = checkmark if not self.last_extra_text else cross

        print("\n----- Example Prompt-Completion -----")
        print(f"PROMPT: {self.last_prompt}")
        print(f"COMPLETION: {self.last_completion}")
        print(f"OVERALL COMPLETION LENGTH (tokens): {self.last_completion_length}")
        print(f"REWARD: {self.last_reward}")
        print(f"EXTRACTED ANSWER: {self.last_extracted_answer}")
        print(f"GROUND TRUTH: {self.last_ground_truth}")
        print(f"{answer_match_status} ANSWER MATCH: {self.last_answer_match}")
        print(f"{metaphor_status} Includes <metaphor>: {self.last_metaphor_tag}")
        print(f"{reasoning_status} Includes <reasoning>: {self.last_reasoning_tag}")
        print(f"{answer_status} Includes <answer>: {self.last_answer_tag}")
        print(f"{final_answer_status} Includes <final_answer>: {self.last_final_answer_tag}")
        print(f"{extra_text_status} Includes Extra Text: {self.last_extra_text}")
        print("----- End of Example -----\n")

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history and "loss" in state.log_history[-1]:
            loss = state.log_history[-1]["loss"]
            self.running_loss.append(loss)
            print(f"Step {state.global_step}: Loss = {loss:.4f}")
        return control

    def update_rewards(self, rewards):
        self.running_rewards.extend(rewards)

# -----------------------------------------------------------------------------
# Helper function to extract the final answer from the answer text.
# -----------------------------------------------------------------------------
def extract_final_answer(answer_text):
    answer_text = answer_text.strip()
    special_formats = re.findall(r"(?:answer|result|total)(?:\s+is)?\s*[:=]?\s*([-+]?\d*\.?\d+)", answer_text, re.IGNORECASE)
    if special_formats:
        return special_formats[-1]
    numbers = re.findall(r"[-+]?\d*\.?\d+", answer_text)
    if numbers:
        return numbers[-1]
    sentences = re.split(r"[.!?]", answer_text)
    for sentence in reversed(sentences):
        sentence = sentence.strip()
        if sentence:
            return sentence
    return answer_text

# -----------------------------------------------------------------------------
# Helper function to normalize numeric answers for consistency and comparison.
# -----------------------------------------------------------------------------
def normalize_answer(answer):
    answer = answer.strip().replace("$", "").replace(",", "")
    numbers = re.findall(r"[-+]?\d*\.?\d+", answer)
    if numbers:
        return numbers[-1].lstrip("0")
    return answer

# -----------------------------------------------------------------------------
# NEW: Tokenization function to enforce truncation to MAX_SEQ_LENGTH.
# -----------------------------------------------------------------------------
def tokenize_prompt(example, tokenizer):
    prompt_str = " ".join([msg["content"] for msg in example["prompt"]])
    tokenized = tokenizer(
        prompt_str,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_attention_mask=True
    )
    example["input_ids"] = tokenized["input_ids"]
    example["attention_mask"] = tokenized["attention_mask"]
    return example

# ----------------------------------------------------------------------------
# The GRPO reward function!
# ----------------------------------------------------------------------------
def reward_function(completions, ground_truth, prompts=None, **kwargs):
    rewards = []
    total_rewards = []
    detailed_stats = {
        "format_rewards": [],
        "reasoning_rewards": [],
        "metaphor_rewards": [],
        "answer_rewards": [],
        "final_answer_rewards": [],
        "ordering_penalty": [],
        "outside_text_penalty": [],
        "correctness_rewards": [],
        "total": [],
        "includes_reasoning": [],
        "includes_metaphor": [],
        "includes_answer": [],
        "includes_final_answer": [],
        "includes_extra_text": [],
        "completion_lengths": [],
    }
    noise_factor = 0.1

    reasoning_threshold = 1000
    ideal_reasoning_reward = 3.5
    reasoning_decay = 0.005

    metaphor_threshold = 50
    ideal_metaphor_reward = 2.0
    metaphor_decay = 0.1

    answer_threshold = 50
    ideal_answer_reward = 2.0
    answer_decay = 0.1

    ideal_final_answer_reward = 1.0

    for i, (comp, gt) in enumerate(zip(completions, ground_truth)):
        text = comp[0]["content"]
        completion_length = len(text.split())
        detailed_stats["completion_lengths"].append(completion_length)

        reasoning_start = text.find("<reasoning>")
        reasoning_end = text.find("</reasoning>")
        metaphor_start = text.find("<metaphor>")
        metaphor_end = text.find("</metaphor>")
        answer_start = text.find("<answer>")
        answer_end = text.find("</answer>")
        final_answer_start = text.find("<final_answer>")
        final_answer_end = text.find("</final_answer>")

        includes_reasoning = (reasoning_start != -1 and reasoning_end != -1)
        includes_metaphor = (metaphor_start != -1 and metaphor_end != -1)
        includes_answer = (answer_start != -1 and answer_end != -1)
        includes_final_answer = (final_answer_start != -1 and final_answer_end != -1)

        detailed_stats["includes_reasoning"].append(includes_reasoning)
        detailed_stats["includes_metaphor"].append(includes_metaphor)
        detailed_stats["includes_answer"].append(includes_answer)
        detailed_stats["includes_final_answer"].append(includes_final_answer)

        correct_order = (includes_metaphor and includes_reasoning and includes_answer and includes_final_answer and
                         metaphor_start < reasoning_start < answer_start < final_answer_start)
        format_reward = 4 if correct_order else 0.0
        detailed_stats["format_rewards"].append(format_reward)

        includes_extra_text = False
        outside_text_penalty = 0.0
        if includes_reasoning and includes_metaphor and includes_answer and includes_final_answer:
            text_before = text[:metaphor_start].strip()
            text_between1 = text[metaphor_end + len("</metaphor>"):reasoning_start].strip()
            text_between2 = text[reasoning_end + len("</reasoning>"):answer_start].strip()
            text_between3 = text[answer_end + len("</answer>"):final_answer_start].strip()
            text_after = text[final_answer_end + len("</final_answer>"):].strip()
            if text_before or text_between1 or text_between2 or text_between3 or text_after:
                includes_extra_text = True
                penalty_count = sum(bool(txt) for txt in [text_before, text_between1, text_between2, text_between3, text_after])
                outside_text_penalty = -2.0 * penalty_count
        else:
            outside_text_penalty = -2.0

        detailed_stats["outside_text_penalty"].append(outside_text_penalty)
        detailed_stats["includes_extra_text"].append(includes_extra_text)

        metaphor_reward = 0.0
        if includes_metaphor:
            metaphor_text = text[metaphor_start + len("<metaphor>"):metaphor_end].strip()
            metaphor_tokens = metaphor_text.split()
            num_tokens_metaphor = len(metaphor_tokens)
            if num_tokens_metaphor <= metaphor_threshold:
                metaphor_reward = (num_tokens_metaphor / metaphor_threshold) * ideal_metaphor_reward
            else:
                excess = num_tokens_metaphor - metaphor_threshold
                metaphor_reward = ideal_metaphor_reward * math.exp(-metaphor_decay * excess)
        detailed_stats["metaphor_rewards"].append(metaphor_reward)

        reasoning_reward = 0.0
        if includes_reasoning:
            reasoning_text = text[reasoning_start + len("<reasoning>"):reasoning_end].strip()
            reasoning_tokens = reasoning_text.split()
            num_tokens_reasoning = len(reasoning_tokens)
            if num_tokens_reasoning <= reasoning_threshold:
                reasoning_reward = (num_tokens_reasoning / reasoning_threshold) * ideal_reasoning_reward
            else:
                excess = num_tokens_reasoning - reasoning_threshold
                reasoning_reward = ideal_reasoning_reward * math.exp(-reasoning_decay * excess)
        detailed_stats["reasoning_rewards"].append(reasoning_reward)

        answer_reward = 0.0
        if includes_answer:
            answer_text = text[answer_start + len("<answer>"):answer_end].strip()
            answer_tokens = answer_text.split()
            num_tokens_answer = len(answer_tokens)
            if num_tokens_answer <= answer_threshold:
                answer_reward = (num_tokens_answer / answer_threshold) * ideal_answer_reward
            else:
                excess = num_tokens_answer - answer_threshold
                answer_reward = ideal_answer_reward * math.exp(-answer_decay * excess)
        detailed_stats["answer_rewards"].append(answer_reward)

        correctness_reward = 0.0
        predicted_final_answer = "MISSING"
        gt_str = str(gt).strip()
        match = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL)
        if match:
            predicted_final_answer = match.group(1).strip()
        norm_pred = normalize_answer(predicted_final_answer)
        norm_gt = normalize_answer(gt_str)
        if norm_pred == norm_gt:
            correctness_reward = 7.0
        elif norm_gt.isnumeric() and norm_pred in norm_gt:
            correctness_reward = 2.5
        detailed_stats["correctness_rewards"].append(correctness_reward)

        total_reward = (format_reward + reasoning_reward + metaphor_reward +
                        answer_reward + correctness_reward + outside_text_penalty)
        noise = np.random.normal(0, noise_factor)
        total_reward += noise

        detailed_stats["total"].append(total_reward)
        total_rewards.append(total_reward)
        rewards.append(total_reward)

        if i == 0 and prompts is not None and hasattr(trainer, 'callback'):
            prompt_text = prompts[i][-1]["content"] if isinstance(prompts[i], list) else prompts[i]
            trainer.callback.update_example(
                prompt_text, text, reward=total_reward, extracted_answer=predicted_final_answer, ground_truth=gt_str,
                includes_reasoning=includes_reasoning,
                includes_metaphor=includes_metaphor,
                includes_answer=includes_answer,
                includes_final_answer=includes_final_answer,
                includes_extra_text=includes_extra_text,
                completion_length=completion_length
            )

    avg_format = sum(detailed_stats["format_rewards"]) / len(detailed_stats["format_rewards"])
    avg_outside = sum(detailed_stats["outside_text_penalty"]) / len(detailed_stats["outside_text_penalty"])
    avg_total = sum(detailed_stats["total"]) / len(detailed_stats["total"])
    avg_completion_length = sum(detailed_stats["completion_lengths"]) / len(detailed_stats["completion_lengths"])
    avg_correctness = sum(detailed_stats["correctness_rewards"]) / len(detailed_stats["correctness_rewards"])

    reward_std = np.std(total_rewards)

    print(f"🔢 Raw rewards: {[round(r, 2) for r in total_rewards[:8]]}{'...' if len(total_rewards) > 8 else ''}")
    print(f"🔍 Reward Stats - Format: {avg_format:.2f}, Outside Text Penalty: {avg_outside:.2f}, Total: {avg_total:.2f}, StdDev: {reward_std:.4f}")
    print(f"Average Completion Token Count: {avg_completion_length:.2f}")
    print(f"Average Correctness Reward: {avg_correctness:.2f}")

    if hasattr(trainer, 'callback') and hasattr(trainer.callback, 'update_rewards'):
        trainer.callback.update_rewards(rewards)

    return rewards

# -----------------------------------------------------------------------------
# Main training function.
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a GRPO model with optional checkpoint resume")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint to resume training from (optional).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--num_train_steps", type=int, default=DEFAULT_TRAINING_STEPS,
                        help="Number of additional training steps to run. If a checkpoint is provided, these steps will be added to the checkpoint's current step count.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -----------------------------------------------------------------------------
    # Load and preprocess the GSM8K dataset.
    # -----------------------------------------------------------------------------
    dataset = load_dataset("gsm8k", "main", split="train")
    dataset = dataset.map(preprocess_example)

    # -----------------------------------------------------------------------------
    # Load the model and tokenizer early so that we can enforce truncation on the dataset.
    # -----------------------------------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=False,      # Disable fast inference for compatibility
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    # -----------------------------------------------------------------------------
    # Apply truncation to each example by tokenizing the prompt.
    # -----------------------------------------------------------------------------
    dataset = dataset.map(tokenize_prompt, fn_kwargs={"tokenizer": tokenizer})

    print("First two preprocessed examples:")
    for i in range(2):
        print("*******************")
        print(dataset[i])

    # -----------------------------------------------------------------------------
    # Apply QLoRA using unsloth's get_peft_model.
    # -----------------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # -----------------------------------------------------------------------------
    # Configure GRPO training parameters.
    # -----------------------------------------------------------------------------
    training_args = GRPOConfig(
        output_dir="./qwen2.5-3b-grpo",
        logging_steps=1,
        num_generations=GRPO_GROUP_SIZE,
        fp16=False,
        optim="adamw_torch_fused",
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=LEARNING_RATE,
        use_vllm=False,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        max_steps=args.num_train_steps,
        save_steps=SAVE_STEPS,
        save_total_limit=NUM_SAVE_CHECKPOINTS,
        temperature=TEMPERATURE,
        beta=0.01,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        dataloader_num_workers=1,
    )
    
    if args.checkpoint_path is not None:
        state_file = os.path.join(args.checkpoint_path, "trainer_state.json")
        try:
            with open(state_file, "r") as f:
                state_data = json.load(f)
            current_steps = state_data.get("global_step", 0)
            print(f"Resuming training from checkpoint with {current_steps} steps. Training for an additional {args.num_train_steps} steps.")
            training_args.max_steps = current_steps + args.num_train_steps
        except Exception as e:
            print("Warning: Could not determine checkpoint step count from trainer_state.json. Proceeding with num_train_steps as max_steps.")

    # -----------------------------------------------------------------------------
    # Create a detailed logging callback instance.
    # -----------------------------------------------------------------------------
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
        processing_class=tokenizer
    )
    
    trainer.add_callback(callback)
    trainer.callback = callback

    if args.checkpoint_path is not None:
        print("Resuming training from checkpoint:", args.checkpoint_path)
    trainer.train(resume_from_checkpoint=args.checkpoint_path)
    
    trainer.save_model("./qwen2.5-3b-grpo-final")
    print("Training completed. Final model saved to ./qwen2.5-3b-grpo-final")

if __name__ == "__main__":
    main()

