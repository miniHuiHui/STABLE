#!/bin/bash
conda run -n qwen_brickgpt python src/brickgpt/prepare_finetuning_dataset_pc.py --input_path AvaLovelace/StableText2Brick --output_path datasets_pc

conda run -n qwen_brickgpt_2 ./scripts/finetune_qwen_pc.zsh Qwen/Qwen3.5-2B ./finetuned_models qwen_pc_model datasets_pc
