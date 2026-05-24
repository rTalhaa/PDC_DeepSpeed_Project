# DeepSpeed ZeRO-2 Fair Run

This folder contains the dual-GPU DeepSpeed implementation used in the project comparison between single-GPU QLoRA and multi-GPU full-parameter fine-tuning.

## What It Runs

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | `HuggingFaceH4/CodeAlpaca_20K` |
| Samples | 1,000 total |
| Split | 900 train / 100 eval |
| Max sequence length | 512 |
| Method | DeepSpeed ZeRO-2 full fine-tuning |
| Precision | FP16 |
| Hardware used | 2 x NVIDIA RTX A4000 |

## Files

| File | Purpose |
|---|---|
| `train_deepspeed_zero2_fair.py` | Main Hugging Face Trainer + DeepSpeed training script |
| `ds_zero2_config.json` | ZeRO-2 runtime configuration |
| `ds_zero2_config_explained.md` | Plain-English explanation of the DeepSpeed config |
| `run_zero2_fair.sh` | End-to-end training, logging, and telemetry parsing script |
| `parse_gpu_log.py` | Converts raw `nvidia-smi` logs into per-GPU summary metrics |
| `RUN_COMMANDS.md` | Manual command sequence for running the experiment |

## Quick Start

```bash
pip install -U -r requirements.txt
bash run_zero2_fair.sh
```

The script starts GPU logging, runs the fair ZeRO-2 experiment, stops logging, and writes the parsed telemetry summary.

## Manual Run

```bash
mkdir -p logs results outputs

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 > logs/gpu_log_zero2_fair.csv &
echo $! > logs/gpu_logger.pid

deepspeed --num_gpus=2 train_deepspeed_zero2_fair.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_name HuggingFaceH4/CodeAlpaca_20K \
  --max_samples 1000 \
  --max_seq_length 512 \
  --epochs 1 \
  --output_dir outputs/zero2_fair_run \
  --deepspeed ds_zero2_config.json

kill $(cat logs/gpu_logger.pid)

python parse_gpu_log.py \
  --log logs/gpu_log_zero2_fair.csv \
  --out results/gpu_summary_zero2_fair.csv
```

## Captured Results

| Metric | Value |
|---|---:|
| Train samples | 900 |
| Eval samples | 100 |
| Global steps | 57 |
| Trainable parameters | 494,032,768 |
| Trainable percentage | 100.0000% |
| Training runtime | 247.091 s |
| Manual wall time | 249.900 s |
| Training samples/s | 3.642 |
| Training steps/s | 0.231 |
| Training loss | 0.792 |
| Evaluation loss | 0.616 |
| Peak allocated memory | 5.9008 GB per rank |
| Peak reserved memory | 9.3398 GB per rank |

Detailed metrics are stored in `results/metrics_zero2_fair.json`, and GPU telemetry is summarized in `results/gpu_summary_zero2_fair.csv`.
