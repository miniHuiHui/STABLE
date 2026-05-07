#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH=/usr/local/cuda/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache

source /home/csgrad/cxu26/miniconda3/etc/profile.d/conda.sh
conda activate qwen_brickgpt_2

cd /home/csgrad/cxu26/BrickGPT
accelerate launch --num_processes 4 train_grpo.py
