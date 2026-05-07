from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
dataset = load_dataset("datasets_pc", split="train[:5]")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="messages",
    max_seq_length=128,
)
print("SFTTrainer initialized successfully")
