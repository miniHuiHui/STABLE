#!/bin/bash
# Monitor GRPO training: check every hour, restart if crashed.
# Run: nohup bash monitor_training.sh > monitor.log 2>&1 &

LOG_DIR="/home/csgrad/cxu26/BrickGPT"
TRAIN_LOG="$LOG_DIR/grpo_training_v3.log"
MONITOR_LOG="$LOG_DIR/monitor.log"
INTERVAL=3600

cd "$LOG_DIR" || exit 1

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$MONITOR_LOG"
}

while true; do
    if ! pgrep -f "train_grpo.py" > /dev/null 2>&1; then
        log_msg "Training not running. Starting..."
        export CUDA_HOME=/usr/local/cuda
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export PATH=/usr/local/cuda/bin:$PATH
        export TRITON_CACHE_DIR=/tmp/triton_cache
        source /home/csgrad/cxu26/miniconda3/etc/profile.d/conda.sh
        conda activate qwen_brickgpt_2
        nohup bash "$LOG_DIR/start_training.sh" >> "$TRAIN_LOG" 2>&1 &
        log_msg "Started training PID $!"
    else
        log_msg "Training still running (PID: $(pgrep -f 'train_grpo.py' | head -1))."
    fi
    sleep "$INTERVAL"
done
