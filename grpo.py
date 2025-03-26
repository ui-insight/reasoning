import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL
import torch
import re
import math
import numpy as np
from datasets import load_dataset, Dataset
from transformers import EarlyStoppingCallback, TextStreamer
from trl import GRPOConfig, GRPOTrainer

# Patch FastLanguageModel for GRPO support
PatchFastRL("GRPO", FastLanguageModel)

# Define constants based on hyperparameters
SEQ_LENGTH = 4096
LORA_RANK = 64
LORA_ALPHA = LORA_RANK * 2
BATCH_SIZE = 32
BETA = 0.02  
MAX_STEPS = 3000

# Constants for section lengths
IDEAL_METAPHOR_LENGTH = 100
MAX_METAPHOR_LENGTH = 150
IDEAL_REASONING_LENGTH = 1200
MAX_REASONING_LENGTH = 1600
IDEAL_ANSWER_LENGTH = 100
MAX_ANSWER_LENGTH = 150
MAX_FINAL_ANSWER_LENGTH = 10

# System prompt
SYSTEM_PROMPT = "You are an AI assistant that provides solutions to math problems."

# XML format patterns and tag order for validation
XML_FORMAT = {
    "metaphor": r"<metaphor>(.*?)</metaphor>",
    "reasoning": r"<reasoning>(.*?)</reasoning>",
    "answer": r"<answer>(.*?)</answer>",
    "final_answer": r"<final_answer>(.*?)</final_answer>"
}
CORRECT_TAG_ORDER = ["metaphor", "reasoning", "answer", "final_answer"]

# Function to extract answer from GSM8K dataset
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Function to prepare the GSM8K dataset
def get_gsm8k_dataset(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]}
            ],
            "answer": extract_hash_answer(x["answer"])
        }
    )
    return data

# =============== REWARD FUNCTIONS =============== #

def extract_xml_content(text, tag):
    """Extract content from XML tags."""
    pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1) if match else ""

def extract_final_answer(completion):
    """Extract the final answer from the completion."""
    content = completion[0]["content"]
    final_answer = extract_xml_content(content, "final_answer")
    return final_answer.strip()

def count_tokens(text, tokenizer):
    """Count tokens in text."""
    if not text:
        return 0
    return len(tokenizer.encode(text))

def normalize_answer(answer_text: str) -> str:
    """Normalize an answer for robust comparison."""
    if not answer_text:
        return ""
    cleaned_text = answer_text.strip().lower()
    cleaned_text = re.sub(r'[$£€,]', '', cleaned_text)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', cleaned_text)
    if numbers:
        last_number = numbers[-1]
        try:
            value = float(last_number)
            if value.is_integer():
                return str(int(value))
            else:
                return str(value).rstrip('0').rstrip('.') if '.' in str(value) else str(value)
        except ValueError:
            return last_number
    words = re.findall(r'\b[a-z]+\b', cleaned_text)
    if words:
        return words[-1]
    return cleaned_text

def is_answer_correct(extracted_answer, ground_truth):
    """Check if the answer is correct using robust comparison."""
    norm_predicted = normalize_answer(extracted_answer)
    norm_actual = normalize_answer(ground_truth)
    if norm_predicted == norm_actual:
        return True
    try:
        if abs(float(norm_predicted) - float(norm_actual)) < 1e-6:
            return True
    except ValueError:
        pass
    return False

def logistic_decay(current, ideal, maximum):
    """
    Calculate a reward using logistic decay:
      - Linear growth up to the ideal length
      - Logistic decay from ideal until maximum
      - Zero beyond maximum
    """
    if current > maximum:
        return 0.0
    if current <= ideal:
        return (current / ideal)
    k = 0.05  # Decay rate
    midpoint = (ideal + maximum) / 2
    return 1 / (1 + math.exp(k * (current - midpoint)))

def check_tag_order_and_format(content):
    """
    Check if tags are in the correct order and no extra text exists outside tags.
    Returns:
      is_valid, missing_tags, wrong_order, has_repeated_tags, has_extra_text
    """
    tag_positions = {}
    all_tag_positions = []
    for tag in CORRECT_TAG_ORDER:
        pattern = re.compile(f"<{tag}>.*?</{tag}>", re.DOTALL)
        matches = list(pattern.finditer(content))
        tag_positions[tag] = [(m.start(), m.end(), tag) for m in matches]
        all_tag_positions.extend(tag_positions[tag])
    all_tag_positions.sort()
    missing_tags = [tag for tag in CORRECT_TAG_ORDER if not tag_positions[tag]]
    has_repeated_tags = any(len(positions) > 1 for positions in tag_positions.values())
    wrong_order = False
    if not missing_tags:
        found_order = [tag for _, _, tag in all_tag_positions]
        wrong_order = found_order != CORRECT_TAG_ORDER
    if all_tag_positions:
        all_text_no_ws = re.sub(r'\s', '', content)
        combined_sections = ''.join([content[start:end] for start, end, _ in all_tag_positions])
        combined_no_ws = re.sub(r'\s', '', combined_sections)
        has_extra_text = len(all_text_no_ws) > len(combined_no_ws)
    else:
        has_extra_text = True
    is_valid = not missing_tags and not wrong_order and not has_repeated_tags and not has_extra_text
    return is_valid, missing_tags, wrong_order, has_repeated_tags, has_extra_text

# ----- Reward functions -----

def correctness_reward_func(prompts, completions, answer, tokenizer, **kwargs) -> list[float]:
    """Awards 10 points if the extracted answer matches the ground truth, 0 otherwise."""
    rewards = []
    for completion, gt in zip(completions, answer):
        extracted = extract_final_answer(completion)
        reward = 10.0 if is_answer_correct(extracted, gt) else 0.0
        rewards.append(reward)
    return rewards

def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Checks format by verifying that all four XML tags are present in order,
    without repeated tags or extra text. Penalties:
      - 2.5 points for each missing tag
      - 5 points if tags are in the wrong order
      - 2.5 points for repeated tags
      - 2.5 points for extra text
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        _, missing_tags, wrong_order, has_repeated_tags, has_extra_text = check_tag_order_and_format(content)
        penalty = len(missing_tags) * 2.5
        penalty += 5.0 if wrong_order else 0.0
        penalty += 2.5 if has_repeated_tags else 0.0
        penalty += 2.5 if has_extra_text else 0.0
        reward = max(0, 10.0 - penalty)
        rewards.append(reward)
    return rewards

def section_length_reward_func(completions, tokenizer, **kwargs) -> list[float]:
    """
    Calculates a reward for section lengths (total max of 11 points):
      - Metaphor: 2 points max (ideal 100 tokens)
      - Reasoning: 4 points max (ideal 1200 tokens)
      - Answer: 2 points max (ideal 100 tokens)
      - Final Answer: 2 points if its token count <= MAX_FINAL_ANSWER_LENGTH, else 0.
    Only computed if the XML format is valid.
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        is_valid, _, _, _, _ = check_tag_order_and_format(content)
        if not is_valid:
            rewards.append(0.0)
            continue
        metaphor = extract_xml_content(content, "metaphor")
        reasoning = extract_xml_content(content, "reasoning")
        answer_sec = extract_xml_content(content, "answer")
        final_answer = extract_xml_content(content, "final_answer")
        
        metaphor_tokens = count_tokens(metaphor, tokenizer)
        reasoning_tokens = count_tokens(reasoning, tokenizer)
        answer_tokens = count_tokens(answer_sec, tokenizer)
        final_answer_tokens = count_tokens(final_answer, tokenizer)
        
        metaphor_reward = 2.0 * logistic_decay(metaphor_tokens, IDEAL_METAPHOR_LENGTH, MAX_METAPHOR_LENGTH)
        reasoning_reward = 4.0 * logistic_decay(reasoning_tokens, IDEAL_REASONING_LENGTH, MAX_REASONING_LENGTH)
        answer_reward = 2.0 * logistic_decay(answer_tokens, IDEAL_ANSWER_LENGTH, MAX_ANSWER_LENGTH)
        final_answer_reward = 2.0 if final_answer_tokens <= MAX_FINAL_ANSWER_LENGTH else 0.0
        
        total = metaphor_reward + reasoning_reward + answer_reward + final_answer_reward
        rewards.append(total)
    return rewards

def combined_reward_func(prompts, completions, answer, tokenizer, **kwargs) -> list[float]:
    """
    Combined reward = correctness (10) + format (10) + section length (11).
    Also prints a detailed unified log (for the first prompt-completion pair)
    following the desired format.
    """
    correctness_rewards = correctness_reward_func(prompts, completions, answer, tokenizer, **kwargs)
    format_rewards = format_reward_func(completions, **kwargs)
    section_rewards = section_length_reward_func(completions, tokenizer, **kwargs)
    
    combined_rewards = []
    for i in range(len(completions)):
        total = correctness_rewards[i] + format_rewards[i] + section_rewards[i]
        combined_rewards.append(total)
        
        if i == 0:
            # Extract the "user" prompt from the prompt list (assumes a list of dicts)
            prompt_text = ""
            for msg in prompts[i]:
                if msg["role"] == "user":
                    prompt_text = msg["content"]
                    break
            content = completions[i][0]["content"]
            extracted = extract_final_answer(completions[i])
            gt = answer[i]
            
            # Determine answer match indicator
            green_check = "\033[92m✔\033[0m"
            red_x = "\033[91m✘\033[0m"
            if is_answer_correct(extracted, gt):
                match_line = f"{green_check} ANSWER MATCH: Correct match"
            else:
                match_line = f"{red_x} ANSWER MATCH: No match"
            
            # Check inclusion of XML tags
            includes_metaphor = bool(re.search(r"<metaphor>.*?</metaphor>", content, re.DOTALL))
            includes_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", content, re.DOTALL))
            includes_answer = bool(re.search(r"<answer>.*?</answer>", content, re.DOTALL))
            includes_final_answer = bool(re.search(r"<final_answer>.*?</final_answer>", content, re.DOTALL))
            
            # Also re-run the format check to capture extra text flag
            _, _, _, _, has_extra_text = check_tag_order_and_format(content)
            
            overall_length = count_tokens(content, tokenizer)
            
            print("\n----- Example Prompt-Completion -----")
            print(f"PROMPT: {prompt_text}")
            print(f"COMPLETION: {content}")
            print(f"CORRECTNESS REWARD: {correctness_rewards[i]:.2f}/10.0")
            print(f"FORMAT REWARD: {format_rewards[i]:.2f}/10.0")
            print(f"SECTION LENGTH REWARD: {section_rewards[i]:.2f}/11.0")
            print(f"EXTRACTED ANSWER: {extracted}")
            print(f"GROUND TRUTH: {gt}")
            print(match_line)
            print(f"{green_check if includes_metaphor else red_x} Includes <metaphor>: {includes_metaphor}")
            print(f"{green_check if includes_reasoning else red_x} Includes <reasoning>: {includes_reasoning}")
            print(f"{green_check if includes_answer else red_x} Includes <answer>: {includes_answer}")
            print(f"{green_check if includes_final_answer else red_x} Includes <final_answer>: {includes_final_answer}")
            print(f"{red_x if has_extra_text else green_check} Includes Extra Text: {has_extra_text}")
            print(f"OVERALL COMPLETION LENGTH (tokens): {overall_length}")
            print(f"TOTAL REWARD: {total:.2f}/31.0")
            print("----- End of Example -----\n")
            
    return combined_rewards

# ----- Optional Stats Callback (disabled logging to avoid duplicate prints) -----
class StatsCallback:
    def on_step_end(self, args, state, control, **kwargs):
        pass

# =============== MAIN TRAINING FUNCTION =============== #

def main():
    print("Loading dataset...")
    dataset = get_gsm8k_dataset()
    
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.7,
    )
    
    adapter_path = "./qwen2.5-3b-sft-unsloth-4bit/checkpoint-1000"
    print(f"Loading adapter from {adapter_path}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=True,
    )
    
    model.active_adapter = "default"
    
    from peft import PeftModel
    print(f"Loading adapter weights from {adapter_path}/adapter_model.safetensors")
    model.load_adapter(adapter_path, adapter_name="default")
    
    print("Setting up GRPO configuration...")
    training_args = GRPOConfig(
        use_vllm=True,
        output_dir="qwen2.5-3b-reasoning-model",
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=512,
        max_completion_length=SEQ_LENGTH - 512,
        max_steps=MAX_STEPS,  
        save_steps=100,
        max_grad_norm=0.1,
        report_to="none",
        beta=BETA,
    )
    
    print("Initializing GRPO trainer...")
    def wrapped_combined_reward(prompts, completions, answer, **kwargs):
        return combined_reward_func(prompts, completions, answer, tokenizer, **kwargs)
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[wrapped_combined_reward],
        args=training_args,
        train_dataset=dataset,
        # callbacks=[StatsCallback()],  # Uncomment if desired
    )
    
    print("Starting GRPO training...")
    trainer.train()
    
    print("Training complete. Saving model...")
    model.save_pretrained("qwen2.5-3b-reasoning-model/final")
    tokenizer.save_pretrained("qwen2.5-3b-reasoning-model/final")
    
    print("All done!")

if __name__ == "__main__":
    main()

