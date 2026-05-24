# DeepSpeed ZeRO-2 Fair Test

## 1. Install

```bash
## pip install -U -r requirements.txt
```

## 2. Verify both GPUs

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_name(1))"
```

## 3. Start GPU logging

```bash
mkdir -p logs results outputs
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > logs/gpu_log_zero2_fair.csv &
echo $! > logs/gpu_logger.pid
```

## 4. Run fair DeepSpeed ZeRO-2 test

```bash
deepspeed --num_gpus=2 train_deepspeed_zero2_fair.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_name HuggingFaceH4/CodeAlpaca_20K \
  --max_samples 1000 \
  --max_seq_length 512 \
  --epochs 1 \
  --output_dir outputs/zero2_fair_run \
  --deepspeed ds_zero2_config.json
```

## 5. Stop GPU logging

```bash
kill $(cat logs/gpu_logger.pid)
```

## 6. Parse GPU log

```bash
python parse_gpu_log.py --log logs/gpu_log_zero2_fair.csv --out results/gpu_summary_zero2_fair.csv
```

## 7. Collect these files

```text
results/metrics_zero2_fair.json
results/gpu_summary_zero2_fair.csv
logs/gpu_log_zero2_fair.csv
outputs/zero2_fair_run/train_results.json
outputs/zero2_fair_run/eval_results.json
```
