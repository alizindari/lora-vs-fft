"""
Fine-tune models with LoRA or full fine-tuning.

Two pipelines:
  1. Qwen 2.5 (causal LM) on commonsense_170k → eval with lm-eval-harness
  2. RoBERTa (classification) on GLUE tasks → eval on task dev set

Usage:
  # Qwen causal LM on commonsense_170k
  python train.py --model_name Qwen/Qwen2.5-3B --data_path data/commonsense_170k.json ...

  # RoBERTa classification on a GLUE task
  python train.py --model_name roberta-base --glue_task sst2 ...
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import spearmanr
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# GLUE task configuration
# ---------------------------------------------------------------------------

GLUE_TASKS = {
    "sst2":  {"num_labels": 2, "keys": ("sentence", None)},
    "cola":  {"num_labels": 2, "keys": ("sentence", None)},
    "mrpc":  {"num_labels": 2, "keys": ("sentence1", "sentence2")},
    "stsb":  {"num_labels": 1, "keys": ("sentence1", "sentence2")},
    "qnli":  {"num_labels": 2, "keys": ("question", "sentence")},
    "rte":   {"num_labels": 2, "keys": ("sentence1", "sentence2")},
    "mnli":  {"num_labels": 3, "keys": ("premise", "hypothesis")},
}


# ---------------------------------------------------------------------------
# Pipeline 1: Qwen causal LM on commonsense_170k
# ---------------------------------------------------------------------------

def tokenize_with_mask(example, tokenizer, max_length):
    """Tokenize instruction + output, mask instruction tokens in labels."""
    instruction = example["instruction"].strip()
    output = example["output"].strip()

    prompt_ids = tokenizer(instruction + "\n", add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(output + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def load_commonsense_data(data_path, tokenizer, max_length, max_samples, val_size, seed):
    """Load commonsense_170k JSON, tokenize with answer-only loss, and split."""
    with open(data_path) as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} examples from {data_path}")

    ds = Dataset.from_list(raw)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    ds = ds.map(
        lambda ex: tokenize_with_mask(ex, tokenizer, max_length),
        remove_columns=ds.column_names,
    )

    split = ds.train_test_split(test_size=val_size, seed=seed)
    return split["train"], split["test"]


def train_causal_lm(args):
    """Qwen causal LM fine-tuning pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        pretrained_model_name_or_path=args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if args.finetuning_type == "lora" and args.deepspeed is None:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.finetuning_type == "lora":
        target_modules = args.lora_target
        if target_modules != "all-linear":
            target_modules = [m.strip() for m in target_modules.split(",")]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_ds, eval_ds = load_commonsense_data(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.cutoff_len,
        max_samples=args.max_samples,
        val_size=args.val_size,
        seed=args.data_seed,
    )
    print(f"Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    optim_name = "adamw_torch" if args.optimizer == "adamw" else "sgd"

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
        max_grad_norm=args.max_grad_norm,
        optim=optim_name,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        seed=args.seed,
        ddp_timeout=180000000,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
        report_to="none",
    )

    trainer = CustomTrainer(
        sgd_momentum=args.sgd_momentum,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


# ---------------------------------------------------------------------------
# Pipeline 2: RoBERTa classification on GLUE
# ---------------------------------------------------------------------------

def load_glue_data(task_name, tokenizer, max_length, max_samples, seed):
    """Load a GLUE task, tokenize, return train and eval datasets."""
    cfg = GLUE_TASKS[task_name]
    key1, key2 = cfg["keys"]

    ds = load_dataset("glue", task_name)
    train_ds = ds["train"]
    # MNLI has matched/mismatched validation; use matched
    eval_key = "validation_matched" if task_name == "mnli" else "validation"
    eval_ds = ds[eval_key]

    if max_samples is not None and max_samples < len(train_ds):
        train_ds = train_ds.shuffle(seed=seed).select(range(max_samples))

    def tokenize_fn(examples):
        if key2 is None:
            return tokenizer(examples[key1], truncation=True, max_length=max_length)
        return tokenizer(examples[key1], examples[key2], truncation=True, max_length=max_length)

    remove_cols = [c for c in train_ds.column_names if c != "label"]
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    remove_cols = [c for c in eval_ds.column_names if c != "label"]
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)

    return train_ds, eval_ds


def glue_compute_metrics(task_name):
    """Return a compute_metrics function for the given GLUE task."""
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if task_name == "stsb":
            # Regression — Spearman correlation
            preds = preds.squeeze()
            return {"spearman": float(spearmanr(preds, labels).correlation)}
        else:
            preds = np.argmax(preds, axis=-1)
            metrics = {"accuracy": float(accuracy_score(labels, preds))}
            if task_name == "cola":
                metrics["matthews_correlation"] = float(matthews_corrcoef(labels, preds))
            if task_name == "mrpc":
                metrics["f1"] = float(f1_score(labels, preds))
            return metrics
    return compute_metrics


def train_classification(args):
    """RoBERTa sequence classification fine-tuning pipeline."""
    task_name = args.glue_task
    cfg = GLUE_TASKS[task_name]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=cfg["num_labels"],
    )

    if args.finetuning_type == "lora":
        target_modules = args.lora_target
        if target_modules != "all-linear":
            target_modules = [m.strip() for m in target_modules.split(",")]

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=["classifier"],  # always train classification head
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_ds, eval_ds = load_glue_data(
        task_name=task_name,
        tokenizer=tokenizer,
        max_length=args.cutoff_len,
        max_samples=args.max_samples,
        seed=args.data_seed,
    )
    print(f"Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    optim_name = "adamw_torch" if args.optimizer == "adamw" else "sgd"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        optim=optim_name,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="spearman" if task_name == "stsb" else
                              "matthews_correlation" if task_name == "cola" else "accuracy",
        seed=args.seed,
        report_to="none",
    )

    trainer = CustomTrainer(
        sgd_momentum=args.sgd_momentum,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=glue_compute_metrics(task_name),
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


# ---------------------------------------------------------------------------
# Custom Trainer for SGD with momentum
# ---------------------------------------------------------------------------

class CustomTrainer(Trainer):
    """Trainer subclass that supports SGD with configurable momentum."""

    def __init__(self, sgd_momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.sgd_momentum = sgd_momentum

    def create_optimizer(self):
        if self.args.optim == "sgd":
            decay_params = []
            no_decay_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(nd in name for nd in ("bias", "LayerNorm", "layer_norm", "rmsnorm")):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            self.optimizer = torch.optim.SGD(
                [
                    {"params": decay_params, "weight_decay": self.args.weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=self.args.learning_rate,
                momentum=self.sgd_momentum,
            )
            return self.optimizer

        return super().create_optimizer()


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen or RoBERTa models")

    # Model
    p.add_argument("--model_name", type=str, required=True,
                   help="HuggingFace model name (e.g. Qwen/Qwen2.5-3B, roberta-base)")

    # Method
    p.add_argument("--finetuning_type", type=str, choices=["full", "lora"], default="full")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target", type=str, default="all-linear",
                   help="LoRA targets: 'all-linear' or comma-separated. "
                        "For RoBERTa 'all-linear' defaults to 'query,value'")

    # Dataset — mutually exclusive: --glue_task (RoBERTa) or --data_path (Qwen)
    p.add_argument("--glue_task", type=str, default=None, choices=list(GLUE_TASKS.keys()),
                   help="GLUE task name for RoBERTa classification")
    p.add_argument("--data_path", type=str, default="data/commonsense_170k.json",
                   help="Path to training JSON for Qwen causal LM")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--cutoff_len", type=int, default=512)
    p.add_argument("--val_size", type=float, default=0.2)

    # Training
    p.add_argument("--num_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Optimizer
    p.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--sgd_momentum", type=float, default=0.9)

    # Logging
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=40,
                   help="Eval every N steps (Qwen only; RoBERTa evals every epoch)")
    p.add_argument("--save_steps", type=int, default=40,
                   help="Save every N steps (Qwen only; RoBERTa saves every epoch)")

    # Output
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=42,
                   help="Seed for data subsampling and train/val split (fix across methods for fair comparison)")

    # DeepSpeed (Qwen only)
    p.add_argument("--deepspeed", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    is_glue = args.glue_task is not None

    # ---- Print config ----
    print("=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Model:            {args.model_name}")
    if is_glue:
        print(f"Pipeline:         classification (GLUE)")
        print(f"Task:             {args.glue_task}")
    else:
        print(f"Pipeline:         causal LM (commonsense)")
        print(f"Dataset:          {args.data_path}")
    print(f"Max samples:      {args.max_samples or 'all'}")
    print(f"Method:           {args.finetuning_type}")
    if args.finetuning_type == "lora":
        print(f"LoRA rank:        {args.lora_rank}")
        print(f"LoRA alpha:       {args.lora_alpha}")
        print(f"LoRA dropout:     {args.lora_dropout}")
        print(f"LoRA target:      {args.lora_target}")
    print(f"Optimizer:        {args.optimizer}" +
          (f" (momentum={args.sgd_momentum})" if args.optimizer == "sgd" else ""))
    print(f"LR:               {args.learning_rate}")
    print(f"Epochs:           {args.num_epochs}")
    print(f"Weight decay:     {args.weight_decay}")
    print(f"Batch size/dev:   {args.per_device_batch}")
    print(f"Data seed:        {args.data_seed}")
    print(f"Train seed:       {args.seed}")
    print(f"Output:           {args.output_dir}")
    print("=" * 50)

    # ---- Save run config ----
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Dispatch ----
    if is_glue:
        train_classification(args)
    else:
        train_causal_lm(args)

    print(f"\nModel saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
