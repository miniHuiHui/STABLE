import os
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3.5-2B")
    dataset_name: str = field(default="datasets_pc")
    output_dir: str = field(default="./finetuned_models/qwen_pc_model")
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=2e-4)

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype="auto", trust_remote_code=True)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    
    dataset = load_dataset(args.dataset_name)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        eval_steps=250,
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        save_steps=500,
        logging_steps=10,
        bf16=True,
        dataset_text_field="messages",
        max_length=20480,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else None,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
