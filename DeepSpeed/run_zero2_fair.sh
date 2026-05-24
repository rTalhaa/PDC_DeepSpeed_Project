#!/bin/bash

set -e

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_TIMEOUT=1800
export CUDA_LAUNCH_BLOCKING=0

mkdir -p logs results outputs
echo "Starting GPU logger..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > logs/gpu_log_zero2_fair.csv &
LOGGER_PID=$!
echo $LOGGER_PID > logs/gpu_logger.pid

echo "Starting DeepSpeed ZeRO-2 fair run..."

deepspeed --num_gpus=2 train_deepspeed_zero2_fair.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_name HuggingFaceH4/CodeAlpaca_20K \
  --max_samples 1000 \
  --max_seq_length 512 \
  --epochs 1 \
  --output_dir outputs/zero2_fair_run \
  --deepspeed ds_zero2_config.json 2>&1 | tee logs/train_zero2_fair_terminal.log

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo "Stopping GPU logger..."
kill $LOGGER_PID || true

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
  echo "Training failed with exit code $TRAIN_EXIT_CODE"
  echo "Check logs/train_zero2_fair_terminal.log"
  exit $TRAIN_EXIT_CODE
fi

echo "Parsing GPU log..."
python parse_gpu_log.py \
  --log logs/gpu_log_zero2_fair.csv \
  --out results/gpu_summary_zero2_fair.csv

echo "Training completed successfully."
echo "Check:"
echo "results/metrics_zero2_fair.json"
echo "results/gpu_summary_zero2_fair.csv"
echo "logs/train_zero2_fair_terminal.log"
