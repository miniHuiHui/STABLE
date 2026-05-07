import subprocess
import sys

cmd = [
    "accelerate", "launch", "--num_processes", "4",
    "-m", "trl.commands.scripts.sft",
    "--model_name_or_path", "Qwen/Qwen3.5-2B",
    "--dataset_name", "datasets_pc",
    "--dataset_text_field", "messages",
    "--output_dir", "./finetuned_models/qwen_pc_model",
]

result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", result.stdout[-2000:])
print("STDERR:", result.stderr[-2000:])
