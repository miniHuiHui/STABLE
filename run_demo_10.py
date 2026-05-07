import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append('src')
from brickgpt.data.brick_structure import BrickStructure

def plot_voxels(points, index):
    max_x, max_y, max_z = points.max(axis=0)
    voxels = np.zeros((max_x + 1, max_y + 1, max_z + 1), dtype=bool)
    for x, y, z in points:
        voxels[x, y, z] = True
        
    colors = np.empty(voxels.shape, dtype=object)
    cmap = plt.get_cmap('viridis')
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            for z in range(max_z + 1):
                if voxels[x, y, z]:
                    norm_z = z / max_z if max_z > 0 else 0
                    rgba = cmap(norm_z)
                    colors[x, y, z] = rgba

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=0.5, alpha=0.9)
    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect((max_x, max_y, max_z))
    
    output_dir = "demo_3996_outputs"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/input_{index}.png", bbox_inches='tight')
    plt.close(fig)

model_path = "Qwen/Qwen3.5-2B"
peft_model_id = "finetuned_models/qwen_pc_model/checkpoint-3996"
dataset_path = "datasets_pc/test.jsonl"
output_dir = "demo_3996_outputs"
os.makedirs(output_dir, exist_ok=True)

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

print("Loading LoRA weights from checkpoint 3996...")
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

print("Processing 10 samples...")
with open(dataset_path, 'r') as f:
    for i in range(10):
        sample_line = f.readline()
        if not sample_line:
            break
        sample = json.loads(sample_line)

        messages = sample["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        ground_truth = next((m["content"] for m in messages if m["role"] == "assistant"), None)
        
        # Plot input point cloud
        pc_str = prompt_messages[1]["content"].split('### Input Point Cloud:\n')[1].strip()
        points = []
        for p in pc_str.split('), ('):
            p = p.replace('(', '').replace(')', '')
            try:
                x, y, z = map(int, p.split(','))
                points.append((x, y, z))
            except ValueError:
                pass
        if points:
            plot_voxels(np.array(points), i+1)

        text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        print(f"Generating for sample {i+1}...")
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=8192,
                do_sample=False
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Save output
        out_txt = f"{output_dir}/sample_{i+1}_gen.txt"
        with open(out_txt, "w") as out_f:
            out_f.write(response.strip())
            
        # Try convert to LDR
        try:
            t = response.strip().split("\n")
            t = [x for x in t if x.endswith(")")]
            if t:
                s = BrickStructure.from_txt("\n".join(t))
                with open(f"{output_dir}/sample_{i+1}_gen.ldr", "w") as out_f:
                    out_f.write(s.to_ldr())
        except Exception as e:
            print(f"Failed to create LDR for sample {i+1}: {e}")

print("Done! All results saved in", output_dir)
