#!/bin/bash
set -euo pipefail

cd /home/cloud/openpi
source .venv/bin/activate

# export CUDA_VISIBLE_DEVICES=4,5,6,7 
export FORCE_TORQUE_HORIZON=1

# wandb: offline if no API key provided
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
else
    unset WANDB_MODE
fi

LOG_DIR=/home/cloud/logs
LOG_TS=$(date +%Y%m%d_%H%M%S)
STDOUT_LOG="${LOG_DIR}/${LOG_TS}.o"
STDERR_LOG="${LOG_DIR}/${LOG_TS}.e"
mkdir -p "$LOG_DIR"
exec >"$STDOUT_LOG" 2>"$STDERR_LOG"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

# shellcheck disable=SC2086
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="8" \
    /home/cloud/openpi/scripts/train_pytorch.py pi0_ur3_robotiq_ft \
    --exp_name "jarvis_one" \
                --batch_size 128 \
                        --num_workers 16 \
                --no-pytorch-gradient-checkpointing \
                        --freeze_pretrained_steps 0 \
                        --num_train_steps 40000 \
                --save_interval 10000 \
                --pytorch_weight_path "/home/cloud/model/pi0_base_pytorch/"