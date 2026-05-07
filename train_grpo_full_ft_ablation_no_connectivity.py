"""
Ablation: collision + shape + interlocking (no connectivity).
Full fine-tuning GRPO training script for BrickGPT.
"""
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl_vllm_qwen35_patch import (
    patch_grpo_deepspeed_model_device_mismatch,
    patch_vllm_generation_qwen35_prefixes,
)

sys.path.insert(0, "src")
from reward_fn import (
    collision_reward,
    interlocking_reward,
    shape_reward,
)

MODEL_NAME = os.getenv("MODEL_NAME")
INIT_MODEL_PATH = os.getenv(
    "INIT_MODEL_PATH",
    MODEL_NAME or "finetuned_models/qwen35_sft_full_ft_run1_official",
)
DATASET_DIR = os.getenv("DATASET_DIR", "datasets_pc")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "finetuned_models/qwen_pc_grpo_full_ft_ablation_no_connectivity")
RUN_NAME = os.getenv("RUN_NAME", "brickgpt-grpo-full-ft-ablation-no-connectivity")
USE_WANDB = os.getenv("USE_WANDB", "1").lower() not in {"0", "false", "no"}
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2")


def load_initial_model():
    return AutoModelForCausalLM.from_pretrained(
        INIT_MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPLEMENTATION,
    )


def get_latest_checkpoint(output_dir: str) -> str | None:
    checkpoints = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: p.name)
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def main():
    if USE_WANDB:
        os.environ.setdefault("WANDB_PROJECT", os.getenv("WANDB_PROJECT", "brickgpt-grpo"))
        if os.getenv("WANDB_ENTITY"):
            os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
        if os.getenv("WANDB_MODE"):
            os.environ["WANDB_MODE"] = os.getenv("WANDB_MODE")
        if os.getenv("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

    tokenizer = AutoTokenizer.from_pretrained(INIT_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_initial_model()

    dataset = load_dataset("json", data_files={
        "train": f"{DATASET_DIR}/train_grpo.jsonl",
    })

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        num_generations=8,
        max_completion_length=8192,
        temperature=0.7,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed="ds_config_zero2.json",
        logging_steps=1,
        save_steps=500,
        save_total_limit=3,
        report_to="wandb" if USE_WANDB else "none",
        run_name=RUN_NAME,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            collision_reward,
            shape_reward,
            interlocking_reward,
        ],
        args=grpo_config,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )
    patch_vllm_generation_qwen35_prefixes(trainer)
    patch_grpo_deepspeed_model_device_mismatch(trainer)

    resume_checkpoint = get_latest_checkpoint(OUTPUT_DIR)
    if resume_checkpoint is not None:
        print(f"Resuming GRPO full FT (no-connectivity ablation) from {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"GRPO full FT (no-connectivity ablation) saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
