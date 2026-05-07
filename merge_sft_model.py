"""Merge SFT LoRA adapter into base model and save to disk."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-2B"
SFT_ADAPTER = "finetuned_models/qwen_pc_model/checkpoint-3996"
OUTPUT_DIR = "/data/cxu26/finetuned_models/qwen_pc_sft_merged"

print(f"Loading base model: {MODEL_NAME}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
)

print(f"Loading SFT adapter: {SFT_ADAPTER}")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER)

print("Merging and saving...")
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Merged model saved to {OUTPUT_DIR}")
