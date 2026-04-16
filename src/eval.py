"""
Evaluate fine-tuned models.

Two pipelines:
  1. Qwen (causal LM): lm-eval-harness on 8 commonsense benchmarks
  2. RoBERTa (classification): evaluate on GLUE task dev set

Usage:
  # Qwen — eval with lm-eval-harness
  python eval.py --model_path saves/qwen2.5-3b/.../checkpoint-500
  python eval.py --base_model Qwen/Qwen2.5-3B --lora_path .../checkpoint-500

  # RoBERTa — eval on GLUE dev set
  python eval.py --model_path saves/roberta-base/sst2/.../best --glue_task sst2
  python eval.py --base_model roberta-base --lora_path .../best --glue_task sst2
"""

import argparse
import json
import os
import tempfile

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

import lm_eval

# Same GLUE config as train.py
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
# LoRA merge helper (for causal LM models)
# ---------------------------------------------------------------------------

def merge_lora_causal(base_model_name, lora_path, output_path):
    """Load base causal LM, merge LoRA adapter, save."""
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.float16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Pipeline 1: Qwen eval via lm-eval-harness
# ---------------------------------------------------------------------------

def eval_causal_lm(model_path, tasks, batch_size, num_fewshot, output_path):
    """Run lm-eval-harness on a causal LM."""
    task_list = [t.strip() for t in tasks.split(",")]

    print("=" * 50)
    print("EVALUATION — causal LM (lm-eval-harness)")
    print("=" * 50)
    print(f"Model:        {model_path}")
    print(f"Tasks:        {task_list}")
    print(f"Batch size:   {batch_size}")
    print(f"Num fewshot:  {num_fewshot}")
    print("=" * 50)

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=float16",
        tasks=task_list,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
    )

    accuracies = {}
    for task_name, task_results in results["results"].items():
        accuracies[task_name] = {
            "acc": task_results.get("acc,none"),
            "acc_norm": task_results.get("acc_norm,none"),
        }

    print("\n" + "=" * 50)
    print("RESULTS (accuracy)")
    print("=" * 50)
    for task_name, metrics in sorted(accuracies.items()):
        acc_str = f"{metrics['acc']:.4f}" if metrics["acc"] is not None else "n/a"
        norm_str = f"{metrics['acc_norm']:.4f}" if metrics["acc_norm"] is not None else "n/a"
        print(f"  {task_name:20s}  acc={acc_str}  acc_norm={norm_str}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_data = {
            "accuracies": accuracies,
            "config": {"model": model_path, "tasks": task_list,
                       "batch_size": batch_size, "num_fewshot": num_fewshot},
        }
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Pipeline 2: RoBERTa eval on GLUE dev set
# ---------------------------------------------------------------------------

def eval_classification(model_path, base_model, lora_path, glue_task, batch_size, output_path):
    """Evaluate a classification model on the GLUE task dev set."""
    cfg = GLUE_TASKS[glue_task]
    key1, key2 = cfg["keys"]

    print("=" * 50)
    print("EVALUATION — classification (GLUE dev set)")
    print("=" * 50)
    print(f"Task:         {glue_task}")
    print(f"Model:        {model_path or base_model}")
    if lora_path:
        print(f"LoRA:         {lora_path}")
    print("=" * 50)

    # Load model
    if lora_path and base_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=cfg["num_labels"],
        )
        model = PeftModel.from_pretrained(model, lora_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    elif model_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=cfg["num_labels"],
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=cfg["num_labels"],
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load eval data
    ds = load_dataset("glue", glue_task)
    eval_key = "validation_matched" if glue_task == "mnli" else "validation"
    eval_ds = ds[eval_key]

    def tokenize_fn(examples):
        if key2 is None:
            return tokenizer(examples[key1], truncation=True, max_length=512)
        return tokenizer(examples[key1], examples[key2], truncation=True, max_length=512)

    remove_cols = [c for c in eval_ds.column_names if c != "label"]
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Run inference
    trainer = Trainer(model=model, data_collator=collator)
    predictions = trainer.predict(eval_ds)
    preds = predictions.predictions
    labels = predictions.label_ids

    # Compute metrics
    results = {}
    if glue_task == "stsb":
        preds = preds.squeeze()
        results["spearman"] = float(spearmanr(preds, labels).correlation)
    else:
        preds = np.argmax(preds, axis=-1)
        results["accuracy"] = float(accuracy_score(labels, preds))
        if glue_task == "cola":
            results["matthews_correlation"] = float(matthews_corrcoef(labels, preds))
        if glue_task == "mrpc":
            results["f1"] = float(f1_score(labels, preds))

    # MNLI mismatched
    if glue_task == "mnli":
        mm_ds = ds["validation_mismatched"]
        mm_ds = mm_ds.map(tokenize_fn, batched=True, remove_columns=mm_ds.column_names)
        mm_preds = trainer.predict(mm_ds)
        mm_pred_labels = np.argmax(mm_preds.predictions, axis=-1)
        results["accuracy_mismatched"] = float(accuracy_score(mm_preds.label_ids, mm_pred_labels))

    # Print results
    print("\n" + "=" * 50)
    print(f"RESULTS — {glue_task}")
    print("=" * 50)
    for metric, value in sorted(results.items()):
        print(f"  {metric:25s}  {value:.4f}")

    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_data = {
            "task": glue_task,
            "results": results,
            "config": {
                "model": model_path or base_model,
                "lora_path": lora_path,
                "batch_size": batch_size,
            },
        }
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned models")

    # Model source
    p.add_argument("--model_path", type=str, default=None,
                   help="Path to fine-tuned checkpoint directory")
    p.add_argument("--base_model", type=str, default=None,
                   help="HuggingFace base model name (required for LoRA)")
    p.add_argument("--lora_path", type=str, default=None,
                   help="Path to LoRA adapter checkpoint")

    # Pipeline selection
    p.add_argument("--glue_task", type=str, default=None, choices=list(GLUE_TASKS.keys()),
                   help="If set, run GLUE classification eval instead of lm-eval")

    # Qwen eval settings
    p.add_argument("--tasks", type=str,
                   default="arc_easy,arc_challenge,boolq,commonsense_qa,piqa,winogrande,hellaswag,openbookqa",
                   help="lm-eval task names (Qwen only)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_fewshot", type=int, default=0)

    # Output
    p.add_argument("--output_path", type=str, default=None)

    args = p.parse_args()

    if args.glue_task:
        # ---- RoBERTa GLUE eval ----
        eval_classification(
            model_path=args.model_path,
            base_model=args.base_model,
            lora_path=args.lora_path,
            glue_task=args.glue_task,
            batch_size=args.batch_size,
            output_path=args.output_path,
        )
    else:
        # ---- Qwen lm-eval ----
        if args.lora_path:
            if not args.base_model:
                raise ValueError("--base_model required with --lora_path")
            merge_dir = tempfile.mkdtemp(prefix="lora_merged_")
            model_path = merge_lora_causal(args.base_model, args.lora_path, merge_dir)
        elif args.model_path:
            model_path = args.model_path
        elif args.base_model:
            model_path = args.base_model
        else:
            raise ValueError("Provide --model_path, --lora_path + --base_model, or --base_model")

        eval_causal_lm(
            model_path=model_path,
            tasks=args.tasks,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            output_path=args.output_path,
        )

    print("Done.")


if __name__ == "__main__":
    main()
