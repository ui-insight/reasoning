import os
import json
import argparse
import time
import re
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate synthetic GSM8K dataset with formatted answers")
parser.add_argument("--test", action="store_true", help="Run in test mode (process only 10 examples)")
parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers (default: 8)")
parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for format issues (default: 3)")
args = parser.parse_args()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE"))

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET_NAME = "gsm8k"         # GSM8K dataset name in the datasets library.
DATASET_SPLIT = "train"        # Which split to use.
OUTPUT_FILE = "synthetic_gsm8k_formatted2.jsonl"  # Output file for the synthetic dataset.
MODEL_NAME = "gpt-4o-mini"     # OpenAI model name.
MAX_TOKENS = 1000              # Maximum number of tokens to generate.
TEMPERATURE = 0.7              # Sampling temperature.
NUM_WORKERS = args.workers     # Number of concurrent workers
MAX_RETRIES = args.max_retries # Maximum number of retries for format issues

# -----------------------------------------------------------------------------
# Prompt Template
# -----------------------------------------------------------------------------
prompt_template = (
    "Rewrite the following answer into a fully formatted answer using these tags exactly: "
    "<metaphor>, <reasoning>, <answer>, and <final_answer>. The answer MUST start with <metaphor> and end with </final_answer>.\n"
    "Follow this exact format:\n"
    "<metaphor>...</metaphor>\n"
    "<reasoning>...</reasoning>\n"
    "<answer>...</answer>\n"
    "<final_answer>...</final_answer>\n\n"
    "Question: {question}\n"
    "Original Answer: {original_answer}\n\n"
    "Formatted Answer:"
)

# Modified prompt for retry attempts, emphasizing format requirements
retry_prompt_template = (
    "IMPORTANT: Please rewrite the following answer using EXACTLY these tags in this EXACT order: "
    "<metaphor>, <reasoning>, <answer>, and <final_answer>. Your response MUST start with <metaphor> and end with </final_answer>.\n\n"
    "You MUST include all four tags and ensure they are complete with opening and closing tags.\n"
    "Follow this EXACT format:\n"
    "<metaphor>Brief metaphor related to the problem</metaphor>\n"
    "<reasoning>Step-by-step reasoning process</reasoning>\n"
    "<answer>Calculation steps and numerical answer</answer>\n"
    "<final_answer>Final numerical result in words</final_answer>\n\n"
    "Question: {question}\n"
    "Original Answer: {original_answer}\n\n"
    "Previous attempt had format issues. Please fix and provide a COMPLETE response:"
)

# -----------------------------------------------------------------------------
# Load GSM8K Dataset
# -----------------------------------------------------------------------------
print("Loading GSM8K dataset...")
dataset = load_dataset(DATASET_NAME, name="main", split=DATASET_SPLIT)
print(f"Loaded GSM8K dataset with {len(dataset)} examples.")

# -----------------------------------------------------------------------------
# Format validation functions
# -----------------------------------------------------------------------------
def validate_format(text):
    """
    Validates that the text contains all required tags in the correct order.
    Returns (is_valid, reason) tuple.
    """
    # Check if text contains all required tags
    required_tags = ["<metaphor>", "</metaphor>", "<reasoning>", "</reasoning>", 
                     "<answer>", "</answer>", "<final_answer>", "</final_answer>"]
    
    for tag in required_tags:
        if tag not in text:
            return False, f"Missing tag: {tag}"
    
    # Check tag order
    tag_positions = {
        "<metaphor>": text.find("<metaphor>"),
        "</metaphor>": text.find("</metaphor>"),
        "<reasoning>": text.find("<reasoning>"),
        "</reasoning>": text.find("</reasoning>"),
        "<answer>": text.find("<answer>"),
        "</answer>": text.find("</answer>"),
        "<final_answer>": text.find("<final_answer>"),
        "</final_answer>": text.find("</final_answer>")
    }
    
    # Check if any tags are missing (position = -1)
    for tag, pos in tag_positions.items():
        if pos == -1:
            return False, f"Missing tag: {tag}"
    
    # Check correct order
    if not (tag_positions["<metaphor>"] < tag_positions["</metaphor>"] < 
            tag_positions["<reasoning>"] < tag_positions["</reasoning>"] < 
            tag_positions["<answer>"] < tag_positions["</answer>"] < 
            tag_positions["<final_answer>"] < tag_positions["</final_answer>"]):
        return False, "Tags are not in the correct order"
    
    # Check for truncation
    if not text.endswith("</final_answer>"):
        return False, "Response is truncated"
    
    # Check for content within tags
    pairs = [("<metaphor>", "</metaphor>", 5), 
             ("<reasoning>", "</reasoning>", 5), 
             ("<answer>", "</answer>", 5), 
             ("<final_answer>", "</final_answer>", 1)]  # Only need 1 character for final_answer
    
    for open_tag, close_tag, min_length in pairs:
        start = text.find(open_tag) + len(open_tag)
        end = text.find(close_tag)
        if end - start < min_length:  # Require minimum characters of content based on tag
            return False, f"Insufficient content between {open_tag} and {close_tag}, minimum {min_length} characters required"
    
    return True, "Valid format"

def apply_formatting(text):
    """
    Applies the desired formatting WITHOUT newlines or spaces around tags.
    """
    # First normalize all existing newlines to spaces
    text = text.replace("\\n", " ")  # Replace escaped newlines
    text = text.replace("\n", " ")   # Replace actual newlines
    
    # Clean up any multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")
    
    # Remove spaces between closing and opening tags
    tag_pairs = [
        ("</metaphor>", "<reasoning>"),
        ("</reasoning>", "<answer>"),
        ("</answer>", "<final_answer>")
    ]
    
    for closing_tag, opening_tag in tag_pairs:
        text = text.replace(f"{closing_tag} {opening_tag}", f"{closing_tag}{opening_tag}")
        # Also handle case with multiple spaces
        text = text.replace(f"{closing_tag}  {opening_tag}", f"{closing_tag}{opening_tag}")
    
    return text

# -----------------------------------------------------------------------------
# Helper function to process a single example with retries
# -----------------------------------------------------------------------------
def process_example(example):
    question = example.get("question", "").strip()
    original_answer = example.get("answer", "").strip()
    
    retries = 0
    current_prompt = prompt_template.format(question=question, original_answer=original_answer)
    system_prompt = "You are a helpful assistant that reformats math problem solutions."
    
    while retries <= MAX_RETRIES:
        try:
            # Add a small random delay to prevent rate limiting issues
            time.sleep(0.1)
            
            # Call the OpenAI API to generate the formatted answer using chat completions
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                n=1,
                stop=None
            )
            
            # Extract the generated text from the message content
            generated_text = response.choices[0].message.content.strip()

            # Extract the text starting at <metaphor> and ending at </final_answer>
            start_index = generated_text.find("<metaphor>")
            end_index = generated_text.rfind("</final_answer>")
            
            if start_index != -1 and end_index != -1:
                # Extract just the formatted portion
                formatted_answer = generated_text[start_index : end_index + len("</final_answer>")].strip()
                
                # Validate the format
                is_valid, reason = validate_format(formatted_answer)
                
                if is_valid:
                    # Format is valid, apply the desired formatting and return
                    formatted_answer = apply_formatting(formatted_answer)
                    return {
                        "question": question,
                        "answer": formatted_answer,
                        "success": True,
                        "retries": retries
                    }
                else:
                    # Format is invalid, log and retry
                    if retries < MAX_RETRIES:
                        print(f"Format validation failed on retry {retries}: {reason}. Retrying...")
                        retries += 1
                        # Use more explicit retry prompt
                        current_prompt = retry_prompt_template.format(
                            question=question, 
                            original_answer=original_answer
                        )
                        # Make system prompt more assertive for retries
                        system_prompt = "You are a precise assistant that produces correctly formatted math solutions. Follow the format exactly."
                    else:
                        print(f"Max retries reached for question: {question[:60]}... Last error: {reason}")
                        return {
                            "question": question,
                            "error": f"Format validation failed after {MAX_RETRIES} retries: {reason}",
                            "success": False
                        }
            else:
                # The required tags were not found
                if retries < MAX_RETRIES:
                    print(f"Missing required tags on retry {retries}. Retrying...")
                    retries += 1
                    current_prompt = retry_prompt_template.format(
                        question=question, 
                        original_answer=original_answer
                    )
                    system_prompt = "You are a precise assistant that produces correctly formatted math solutions. Follow the format exactly."
                else:
                    print(f"Max retries reached for question: {question[:60]}... Last error: Missing required tags")
                    return {
                        "question": question,
                        "error": f"Missing required tags after {MAX_RETRIES} retries",
                        "success": False
                    }
                
        except Exception as e:
            print(f"Error processing question: {question[:60]}... Error: {e}")
            return {
                "question": question,
                "error": str(e),
                "success": False
            }

# -----------------------------------------------------------------------------
# Generate Synthetic Examples with Concurrency
# -----------------------------------------------------------------------------
# Limit to 10 examples if in test mode
if args.test:
    dataset = dataset.select(range(min(10, len(dataset))))
    print(f"TEST MODE: Processing only the first {len(dataset)} examples")
    OUTPUT_FILE = "test_" + OUTPUT_FILE

# Convert dataset to a list for processing
examples_to_process = list(dataset)
synthetic_examples = []

print(f"Generating synthetic examples with formatted answers using {NUM_WORKERS} concurrent workers...")
print(f"Maximum retries for format issues: {MAX_RETRIES}")

# Use ThreadPoolExecutor for concurrent processing
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    # Process examples in parallel and track progress with tqdm
    results = list(tqdm(
        executor.map(process_example, examples_to_process),
        total=len(examples_to_process),
        desc="Processing examples"
    ))
    
    # Filter successful results
    synthetic_examples = [result for result in results if result.get("success", False)]
    
    # Report errors and retry statistics
    error_count = sum(1 for result in results if not result.get("success", False))
    retry_stats = {}
    for result in results:
        if result.get("success", False):
            retries = result.get("retries", 0)
            retry_stats[retries] = retry_stats.get(retries, 0) + 1
    
    print(f"\nRetry statistics:")
    for retries, count in sorted(retry_stats.items()):
        print(f"  {retries} retries: {count} examples ({count/len(results)*100:.1f}%)")
    
    if error_count > 0:
        print(f"Encountered {error_count} errors during processing ({error_count/len(results)*100:.1f}%)")

# -----------------------------------------------------------------------------
# Save Synthetic Dataset as JSONL
# -----------------------------------------------------------------------------
print(f"Saving synthetic dataset to {OUTPUT_FILE} ...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in synthetic_examples:
        # Remove the metadata fields before saving
        for field in ["success", "retries"]:
            if field in ex:
                del ex[field]
            
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"Saved {len(synthetic_examples)} examples to {OUTPUT_FILE}.")
