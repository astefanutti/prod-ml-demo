"""Fine-tune Mistral-7B with QLoRA on review summarization/Q&A.

Uses FSDP for multi-node multi-GPU training, designed to run on Slurm
via Kubeflow Trainer's ClusterTrainingRuntime.

Usage (single GPU, testing):
    python training/llm/finetune.py --data-dir data/llm_data --sample

Usage (multi-node FSDP via torchrun):
    torchrun --nnodes=2 --nproc_per_node=4 training/llm/finetune.py \
        --data-dir s3://smartshop/llm_data
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_SEQ_LENGTH = 2048


def load_jsonl_dataset(data_dir: str, split: str) -> HFDataset:
    """Load JSONL instruction-tuning data into a HuggingFace Dataset."""
    split_dir = Path(data_dir) / split
    texts = []

    # Handle both single file and directory of part files (from Spark)
    if split_dir.is_dir():
        for f in sorted(split_dir.glob("*.txt")) + sorted(split_dir.glob("part-*")):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        texts.append(line)
    elif split_dir.with_suffix(".jsonl").exists():
        with open(split_dir.with_suffix(".jsonl")) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    texts.append(line)

    records = []
    for text in texts:
        try:
            record = json.loads(text)
            records.append(record)
        except json.JSONDecodeError:
            continue

    return HFDataset.from_list(records)


def format_instruction(example: dict) -> str:
    """Format an instruction-tuning example into Mistral chat format."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    if input_text:
        prompt = f"[INST] {instruction}\n\n{input_text} [/INST]"
    else:
        prompt = f"[INST] {instruction} [/INST]"

    if output_text:
        return f"{prompt} {output_text}"
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory with train/val/test splits")
    parser.add_argument("--output-dir", default="models/llm-adapter", help="LoRA output dir")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base model name/path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--sample", action="store_true", help="Use small subset for testing")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank == 0:
        print(f"Fine-tuning {args.base_model} with QLoRA")
        print(f"Data: {args.data_dir}")

    # Load datasets
    train_dataset = load_jsonl_dataset(args.data_dir, "train")
    val_dataset = load_jsonl_dataset(args.data_dir, "val")

    if args.sample:
        train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(200, len(val_dataset))))

    if rank == 0:
        print(f"Train examples: {len(train_dataset):,}")
        print(f"Val examples: {len(val_dataset):,}")

    # QLoRA quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if rank == 0:
        model.print_trainable_parameters()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=4,
        # FSDP config for multi-node
        fsdp="full_shard auto_wrap" if int(os.environ.get("WORLD_SIZE", 1)) > 1 else "",
        fsdp_config={
            "fsdp_min_num_params": 1_000_000,
            "fsdp_transformer_layer_cls_to_wrap": "MistralDecoderLayer",
        } if int(os.environ.get("WORLD_SIZE", 1)) > 1 else None,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        formatting_func=format_instruction,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Train
    trainer.train()

    # Save LoRA adapter
    if rank == 0:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
