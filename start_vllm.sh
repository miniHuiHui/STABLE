#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn

source /home/csgrad/cxu26/miniconda3/etc/profile.d/conda.sh
conda activate qwen_brickgpt_2

trl vllm-serve --model Qwen/Qwen3.5-2B --dtype bfloat16 --gpu_memory_utilization 0.7 --max_model_len 4096 --trust_remote_code
