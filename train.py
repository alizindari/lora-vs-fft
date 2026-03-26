"""
Minimal SFT training script for full fine-tuning and LoRA.
Uses HuggingFace Trainer + PEFT directly — no framework needed.
"""

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--finetuning_type", type=str, default="full", choices=["full", "lora"])
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_target", type=str, default="all")
    p.add_argument("--cutoff_len", type=int, default=512)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--num_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=40)
    p.add_argument("--save_steps", type=int, default=40)
    p.add_argument("--deepspeed", type=str, default=None)
    return p.parse_args()


def format_chat(instruction, input_text, output):
    user_content = f"{instruction}\n{input_text}" if input_text else instruction
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def format_prompt(instruction, input_text):
    user_content = f"{instruction}\n{input_text}" if input_text else instruction
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def prepare_dataset(tokenizer, dataset_path, cutoff_len, val_size):
    with open(dataset_path) as f:
        raw = json.load(f)

    records = []
    for ex in raw:
        full_text = format_chat(ex["instruction"], ex.get("input", ""), ex["output"])
        prompt_text = format_prompt(ex["instruction"], ex.get("input", ""))

        full_tok = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=cutoff_len)
        prompt_tok = tokenizer(prompt_text, add_special_tokens=False)
        prompt_len = min(len(prompt_tok["input_ids"]), len(full_tok["input_ids"]))

        labels = [-100] * prompt_len + full_tok["input_ids"][prompt_len:]
        records.append({
            "input_ids": full_tok["input_ids"],
            "attention_mask": full_tok["attention_mask"],
            "labels": labels,
        })

    ds = Dataset.from_list(records)
    split = ds.train_test_split(test_size=val_size, seed=42)
    print(f"Dataset: {len(split['train'])} train, {len(split['test'])} eval")
    return split


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    if args.finetuning_type == "lora":
        target = "all-linear" if args.lora_target == "all" else args.lora_target.split(",")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            target_modules=target,
            lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    split = prepare_dataset(tokenizer, args.dataset_path, args.cutoff_len, args.val_size)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        ddp_timeout=180000000,
        deepspeed=args.deepspeed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
        ),
    )

    trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
