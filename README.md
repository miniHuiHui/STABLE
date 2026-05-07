# STABLE: Stable Brick Generation from Point Clouds

This anonymous repository contains research code and dataset files for point-cloud-conditioned LEGO-style brick generation. The project fine-tunes a causal language model with supervised fine-tuning (SFT), then applies Group Relative Policy Optimization (GRPO) with geometry and stability-oriented rewards.

The repository is prepared for anonymous review and public release. It intentionally excludes local checkpoints, raw experiment logs, W&B run folders, generated outputs, machine-specific paths, and private credentials.

## Contents

- `src/brickgpt/`: brick data structures, parsing, rendering helpers, model wrappers, and stability/connectivity analysis utilities.
- `train_qwen_pc.py`: point-cloud SFT entry point.
- `train_grpo*.py`: GRPO training entry points and ablations.
- `reward_fn.py`: reward functions for shape matching, collision avoidance, connectivity, and interlocking.
- `evaluate_models.py`: unified evaluation script for parse rate, collision metrics, voxel IoU, connectivity, interlocking, and optional physics stability.
- `datasets_pc_parts/` and `datasets_pc_sft_parts/`: line-preserving split JSONL dataset files.
- `scripts/materialize_datasets.py`: recreates the original JSONL dataset layout expected by the training and evaluation scripts.

## Installation

Python 3.10 or newer is recommended. One minimal setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[finetuning]"
```

Optional physics-based stability metrics require a local Gurobi installation and license. Do not commit license files or API tokens. For gated Hugging Face models, set credentials through environment variables such as `HF_TOKEN`.

## Dataset

The full JSONL files are included as split parts so they can be stored in a normal Git repository without Git LFS. Recreate the expected dataset files after cloning:

```bash
python scripts/materialize_datasets.py
```

This creates:

```text
datasets_pc/train.jsonl
datasets_pc/test.jsonl
datasets_pc/train_grpo.jsonl
datasets_pc/test_grpo.jsonl
datasets_pc_sft/train.jsonl
datasets_pc_sft/test.jsonl
```

Each record uses chat-style `messages`. User prompts contain a voxelized point cloud in a fixed `20 x 20 x 20` grid and assistant responses contain brick placements in the format `<brick dimensions> (x,y,z)`.

## Training

SFT example:

```bash
python train_qwen_pc.py
```

GRPO example:

```bash
DATASET_DIR=datasets_pc \
INIT_MODEL_PATH=/path/to/sft/checkpoint \
OUTPUT_DIR=finetuned_models/grpo_run \
python train_grpo_full_ft.py
```

The Slurm scripts are templates. Update `PROJECT_ROOT`, conda paths, partitions, and output directories for your own cluster before use.

## Evaluation

A typical evaluation run is:

```bash
python evaluate_models.py \
  --dataset datasets_pc/test.jsonl \
  --model_type causal_lm \
  --model_path /path/to/model_or_checkpoint \
  --output_dir eval_results/example
```

Enable Gurobi-backed physics metrics only when a valid local solver setup is available.

## Notes

This release builds on public brick-generation tooling and standard open-source libraries. Please respect the licenses of upstream projects and model/data providers when reusing or redistributing this code.
