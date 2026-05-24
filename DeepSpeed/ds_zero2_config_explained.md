# DeepSpeed ZeRO-2 Config Explanation

This file explains `ds_zero2_config.json`. The actual config must stay as valid
JSON, so comments are kept here instead of inside the JSON file.

## Batch Settings

```json
"train_micro_batch_size_per_gpu": "auto",
"train_batch_size": "auto",
"gradient_accumulation_steps": "auto",
"gradient_clipping": "auto"
```

- `train_micro_batch_size_per_gpu`: Number of samples processed by each GPU in
  one forward/backward pass. It is set to `auto`, so Hugging Face `Trainer`
  provides the value from `per_device_train_batch_size`.
- `train_batch_size`: Total effective batch size across GPUs and gradient
  accumulation. It is also handled automatically.
- `gradient_accumulation_steps`: Number of mini-batches accumulated before one
  optimizer update. In the training script this is `8`.
- `gradient_clipping`: Limits large gradients for training stability. `auto`
  lets the Trainer/DeepSpeed integration decide from training arguments.

For this project, the effective training batch size is:

```text
per_device_train_batch_size x number_of_GPUs x gradient_accumulation_steps
= 1 x 2 x 8
= 16
```

## FP16 Mixed Precision

```json
"fp16": {
  "enabled": true,
  "loss_scale": 0,
  "loss_scale_window": 1000,
  "initial_scale_power": 16,
  "hysteresis": 2,
  "min_loss_scale": 1
}
```

- `enabled: true`: Uses FP16 mixed precision to reduce memory usage and improve
  speed on supported GPUs.
- `loss_scale: 0`: Enables dynamic loss scaling. DeepSpeed automatically adjusts
  scaling to avoid FP16 underflow/overflow.
- `loss_scale_window`: How often DeepSpeed considers increasing the loss scale
  after stable steps.
- `initial_scale_power`: Initial dynamic scale is `2^16`.
- `hysteresis`: Number of tolerated overflow checks before reducing the scale.
- `min_loss_scale`: Lowest allowed loss scale.

In simple terms: this section makes training faster and lighter in memory while
DeepSpeed protects numerical stability.

## BF16 Disabled

```json
"bf16": {
  "enabled": false
}
```

BF16 is disabled because the experiment uses FP16. This keeps precision behavior
consistent with the paper and training script.

## ZeRO Optimization

```json
"zero_optimization": {
  "stage": 2,
  "allgather_partitions": true,
  "allgather_bucket_size": 200000000,
  "overlap_comm": true,
  "reduce_scatter": true,
  "reduce_bucket_size": 200000000,
  "contiguous_gradients": true
}
```

- `stage: 2`: Enables ZeRO Stage 2. Optimizer states and gradients are
  partitioned across GPUs. Model parameters are still replicated on each GPU.
- `allgather_partitions: true`: Allows GPUs to gather partitioned data when
  needed during optimization.
- `allgather_bucket_size`: Controls how much data is gathered at once. Larger
  buckets can improve throughput but require more temporary memory.
- `overlap_comm: true`: Overlaps GPU computation with communication between
  GPUs, improving training efficiency.
- `reduce_scatter: true`: Uses reduce-scatter communication to average and
  partition gradients efficiently.
- `reduce_bucket_size`: Controls how much gradient data is reduced at once.
- `contiguous_gradients: true`: Stores gradients in contiguous memory, reducing
  memory fragmentation.

This is the core of the DeepSpeed method used in the paper. ZeRO-2 reduces
memory redundancy for gradients and optimizer states, making full fine-tuning
more practical across two GPUs.

## Optimizer

```json
"optimizer": {
  "type": "AdamW",
  "params": {
    "lr": "auto",
    "betas": "auto",
    "eps": "auto",
    "weight_decay": "auto"
  }
}
```

- `AdamW`: Optimizer commonly used for transformer fine-tuning.
- `lr`, `betas`, `eps`, `weight_decay`: Set to `auto`, so Hugging Face
  `TrainingArguments` supplies the actual values.

In the training script, the learning rate default is:

```text
2e-5
```

## Scheduler

```json
"scheduler": {
  "type": "WarmupDecayLR",
  "params": {
    "warmup_min_lr": "auto",
    "warmup_max_lr": "auto",
    "warmup_num_steps": "auto",
    "total_num_steps": "auto"
  }
}
```

- `WarmupDecayLR`: Starts with a warmup learning rate schedule, then decays.
- All scheduler values are `auto`, so the Trainer calculates them based on
  training length and arguments.

This avoids manually calculating total steps and warmup steps.

## Wall Clock Breakdown

```json
"wall_clock_breakdown": true
```

This asks DeepSpeed to collect timing information for major training operations.
It is useful for performance debugging and understanding where time is spent.

## Short Summary

The config tells DeepSpeed to:

1. Use FP16 mixed precision.
2. Use ZeRO Stage 2.
3. Partition optimizer states and gradients across the two GPUs.
4. Overlap computation with GPU communication.
5. Let Hugging Face Trainer automatically provide batch, optimizer, and
   scheduler values.

That is why the project describes the method as:

```text
Dual-GPU DeepSpeed ZeRO-2 full fine-tuning
```
