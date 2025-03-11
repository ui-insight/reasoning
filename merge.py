#####################################################################################
#
# merge.py
#
# Merge the QLoRA adapators with the original LLM to produce a single new LLM in
# huggingface safetensors b16 format.    This can then be quantized in a later 
# step and saved as a GGUF file for use with Ollama, etc.
#
# Luke Sheneman
# Institute for Interdisciplinary Data Sciences (IIDS)
# March, 2025
# sheneman@uidaho.edu
#
#####################################################################################

import os
import sys
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -----------------------------------------------------------------------------
# Configuration based on your training script
# -----------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # Base model from Hugging Face
ADAPTER_PATH = os.path.expanduser("~/src/reasoning/qwen2.5-3b-grpo/checkpoint-1000")  # QLoRA adapter path
OUTPUT_DIR = "qwen2.5-3b-merged_hf"  # Output directory for merged HF model

# Configure logging for robust diagnostic output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def total_params_size(model):
    """Calculate total size (in GB) of model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    total_size = (param_size + buffer_size) / (1024 ** 3)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Parameter size: {param_size / (1024 ** 3):.2f} GB")
    logger.info(f"Buffer size: {buffer_size / (1024 ** 3):.2f} GB")
    
    return total_size

def load_and_merge_model():
    logger.info(f"Loading base model: {BASE_MODEL}")
    try:
        # Load the base model in BF16 precision
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True
        )
    except Exception as e:
        logger.exception(f"Error loading base model: {e}")
        sys.exit(1)
    
    logger.info("Base model loaded successfully.")
    base_model_size = total_params_size(base_model)
    logger.info(f"Base model size: {base_model_size:.2f} GB")
    
    # Generate a sample with the base model
    test_prompt = "Q: What is the capital of France?\nA:"
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt").to(base_model.device)
        outputs = base_model.generate(**inputs, max_new_tokens=64)
        base_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        logger.info(f"Sample output from base model: {base_output}")
    except Exception as e:
        logger.warning(f"Error generating sample from base model: {e}")
    
    # Load the adapter using PeftModel
    logger.info(f"Loading adapter from: {ADAPTER_PATH}")
    try:
        # Use the standard PeftModel approach to load the adapter
        adapter_model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        logger.info("Adapter loaded successfully.")
    except Exception as e:
        logger.exception(f"Error loading adapter with PeftModel: {e}")
        sys.exit(1)
    
    # Generate a sample with the adapter model
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt").to(adapter_model.device)
        outputs = adapter_model.generate(**inputs, max_new_tokens=64)
        adapter_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        logger.info(f"Sample output with adapter: {adapter_output}")
    except Exception as e:
        logger.warning(f"Error generating sample with adapter model: {e}")
    
    # Merge the adapter weights into the base model
    logger.info("Merging adapter weights into base model...")
    try:
        merged_model = adapter_model.merge_and_unload()
        logger.info("Successfully merged adapter into base model.")
        
        # Check if we still have PEFT attributes
        has_peft_attrs = hasattr(merged_model, "peft_config")
        logger.info(f"Merged model still has PEFT attributes: {has_peft_attrs}")
        
        # Calculate merged model size
        merged_model_size = total_params_size(merged_model)
        logger.info(f"Merged model size: {merged_model_size:.2f} GB")
        
        # Generate with merged model
        try:
            inputs = tokenizer(test_prompt, return_tensors="pt").to(merged_model.device)
            outputs = merged_model.generate(**inputs, max_new_tokens=64)
            merged_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            logger.info(f"Sample output from merged model: {merged_output}")
        except Exception as e:
            logger.warning(f"Error generating sample from merged model: {e}")
        
    except Exception as e:
        logger.exception(f"Error merging adapter weights: {e}")
        sys.exit(1)
    
    return merged_model, tokenizer

# Save the merged model in HuggingFace format.
def save_model_hf(model, tokenizer):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Saving merged model to HuggingFace format at: {OUTPUT_DIR}")
    try:
        # Save the model and tokenizer
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"Successfully saved merged model and tokenizer to {OUTPUT_DIR}")
    except Exception as e:
        logger.exception(f"Error saving merged model: {e}")
        sys.exit(1)
    
    # Log the size of the saved model files
    try:
        total_size = 0
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                if file_size > 10_000_000:  # Log files larger than 10MB
                    logger.info(f"HF model file: {file} - {file_size / (1024 ** 2):.2f} MB")
        logger.info(f"Total HF model size on disk: {total_size / (1024 ** 3):.2f} GB")
    except Exception as e:
        logger.warning(f"Error calculating saved model size: {e}")
    
    # Create a simple README with instructions
    readme_content = f"""# Merged Qwen2.5-3B with QLoRA adapter

This is a merged model of {BASE_MODEL} with QLoRA adapter from {ADAPTER_PATH}.

"""
    
    try:
        with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
            f.write(readme_content)
        logger.info("Created simple README.md")
    except Exception as e:
        logger.warning(f"Error creating README.md: {e}")

def main():
    model, tokenizer = load_and_merge_model()
    save_model_hf(model, tokenizer)
    logger.info(f"Done. Merged model saved to {OUTPUT_DIR}")
    logger.info("You can now use external tools (llama.cpp) to convert this to GGUF and quantize.")

if __name__ == "__main__":
    main()
