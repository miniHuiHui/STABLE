import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

model_path = "Qwen/Qwen3.5-2B"
peft_model_id = "finetuned_models/qwen_pc_model/checkpoint-3000"
dataset_path = "datasets_pc/test.jsonl"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading LoRA weights...")
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

print("Loading sample from test dataset...")
with open(dataset_path, 'r') as f:
    sample_line = f.readline()
    sample = json.loads(sample_line)

messages = sample["messages"]
# prompt is the system and user message
prompt_messages = [m for m in messages if m["role"] != "assistant"]
ground_truth = next((m["content"] for m in messages if m["role"] == "assistant"), None)

text = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize=False,
    add_generation_prompt=True
)
print("Prompt length:", len(text))
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n--- GROUND TRUTH ---")
print(ground_truth[:500] + "..." if len(ground_truth) > 500 else ground_truth)

print("\n--- GENERATED ---")
print(response[:500] + "..." if len(response) > 500 else response)

# Save the generated bricks
output_txt = "demo_checkpoint3000.txt"
with open(output_txt, "w") as f:
    f.write(response.strip())

# Convert to LDR using brick_structure if possible
import sys
sys.path.append('src')
try:
    from brickgpt.data.brick_structure import BrickStructure
    from brickgpt.render_bricks import render_bricks
    
    structure = BrickStructure.from_txt(response.strip())
    ldr_content = structure.to_ldr()
    
    ldr_path = "demo_checkpoint3000.ldr"
    with open(ldr_path, "w") as f:
        f.write(ldr_content)
        
    print(f"Saved LDR to {ldr_path}")
    
    # Try rendering
    png_path = "demo_checkpoint3000.png"
    render_bricks(ldr_path, png_path)
    print(f"Rendered PNG to {png_path}")
    
except Exception as e:
    print("Error processing model output:", e)

print("Done!")
