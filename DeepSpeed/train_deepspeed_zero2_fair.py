import argparse, json, os, time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset_name", default="HuggingFaceH4/CodeAlpaca_20K")
    p.add_argument("--max_samples", type=int, default=5000)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--output_dir", default="outputs/zero2_fair_run")
    p.add_argument("--deepspeed", default="ds_zero2_config.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()

def format_example(example):
    q = str(example["prompt"]).strip()
    a = str(example["completion"]).strip()
    #Adds a fixed reasoning style instruction
    r = "Understand the programming task, identify the required inputs and outputs, select the correct logic or algorithm, and then write the final code solution." 
    return {"text": f"""### Question:
{q}

### Reasoning:
{r}

### Answer:
{a}"""}

def main():
    args = parse_args()
    set_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank in [-1, 0]

    if is_main:
        print("======== DeepSpeed ZeRO-2 Fair Run ========")
        print("Model:", args.model_name)
        print("Dataset:", args.dataset_name)
        print("World size / GPUs:", world_size)
        print("Max samples:", args.max_samples)
        print("Max sequence length:", args.max_seq_length)
        print("Epochs:", args.epochs)

    ds = load_dataset(args.dataset_name)
    
    raw = ds["train"].shuffle(seed=args.seed).select(range(min(args.max_samples, len(ds["train"]))))  #Training split + Shuffle + Max_sample selection
    formatted = raw.map(format_example, remove_columns=raw.column_names)
    split = formatted.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset, eval_dataset = split["train"], split["test"]

    #Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    #Avoideing padding errors during batching
    if tokenizer.pad_token is None:
                                        #use the end-of-sequence token as the pad token
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize(batch):
        out = tokenizer(batch["text"], 
                        truncation=True, #Token > 512 cut them
                        padding="max_length", 
                        max_length=args.max_seq_length)
        
        out["labels"] = out["input_ids"].copy()
        return out

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, trust_remote_code=True)
    model.config.use_cache = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = trainable_params / total_params * 100

    if is_main:
        print("Train size:", len(train_dataset))
        print("Eval size:", len(eval_dataset))
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable percentage: {trainable_pct:.4f}%")

    #Prepare batches for training
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Transformers versions differ: eval_strategy is newer; evaluation_strategy is older.
    try:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            fp16=True,
            bf16=False,
            logging_steps=10,
            save_steps=1000000000,
            eval_strategy="steps",
            eval_steps=100,
            save_total_limit=20000000000000000,
            report_to="none",
            deepspeed=args.deepspeed,
            remove_unused_columns=False,
            max_grad_norm=0.0,
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            fp16=True,
            bf16=False,
            logging_steps=10,
            save_steps=1000000000,
            evaluation_strategy="steps",
            eval_steps=100,
            save_total_limit=20000000000000000,
            report_to="none",
            deepspeed=args.deepspeed,
            remove_unused_columns=False,
            max_grad_norm=0.0,
        )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=collator,
)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    train_result = trainer.train()
    wall_time = time.time() - start
    eval_metrics = trainer.evaluate()

    train_metrics = train_result.metrics
    train_metrics.update({
        "manual_wall_time_sec": wall_time,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "world_size": world_size,
        "model": args.model_name,
        "dataset": args.dataset_name,
        "method": "Dual-GPU DeepSpeed ZeRO-2 Full Fine-Tuning",
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": trainable_pct
    })

    if torch.cuda.is_available():
        train_metrics["rank_peak_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
        train_metrics["rank_peak_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 4)

    combined = {"train_metrics": train_metrics, "eval_metrics": eval_metrics}

    if is_main:
        trainer.save_metrics("train", train_metrics)
        trainer.save_metrics("eval", eval_metrics)
        trainer.save_state()
        Path("results/metrics_zero2_fair.json").write_text(json.dumps(combined, indent=2))
        tokenizer.save_pretrained(Path(args.output_dir) / "tokenizer")
        print("Saved metrics to results/metrics_zero2_fair.json")
        print(json.dumps(combined, indent=2))

if __name__ == "__main__":
    main()
