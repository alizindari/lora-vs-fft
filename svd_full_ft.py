"""
Compute SVD of the weight delta (W_ft - W_base) for full fine-tuning checkpoints.
For each linear layer type (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj),
averages the singular values across all transformer blocks and plots the decay.
"""

import argparse
import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from safetensors import safe_open


LINEAR_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_safetensors_weights(path):
    """Load weights from a single or sharded safetensors file(s) in a directory."""
    weights = {}
    safetensor_files = sorted(
        [f for f in os.listdir(path) if f.endswith(".safetensors") and f.startswith("model")]
    )
    for fname in safetensor_files:
        fpath = os.path.join(path, fname)
        with safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key).float().numpy()
    return weights


def get_layer_keys(weights, module_name):
    """Get all weight keys for a given module type across layers, sorted by layer index."""
    pattern = re.compile(rf"model\.layers\.(\d+)\..+\.{module_name}\.weight")
    matches = []
    for key in weights:
        m = pattern.match(key)
        if m:
            matches.append((int(m.group(1)), key))
    matches.sort(key=lambda x: x[0])
    return [key for _, key in matches]


def compute_svd_for_module(ft_weights, base_weights, module_name):
    """Compute singular values of (W_ft - W_base) for all layers of a given module type."""
    keys = get_layer_keys(ft_weights, module_name)
    if not keys:
        print(f"  Warning: no keys found for {module_name}")
        return None

    all_singular_values = []
    for key in keys:
        delta = ft_weights[key] - base_weights[key]
        sv = np.linalg.svd(delta, compute_uv=False)
        all_singular_values.append(sv)

    # Pad to same length (should already be same for same module type) and average
    max_len = max(len(sv) for sv in all_singular_values)
    padded = np.zeros((len(all_singular_values), max_len))
    for i, sv in enumerate(all_singular_values):
        padded[i, :len(sv)] = sv
    avg_sv = padded.mean(axis=0)

    print(f"  {module_name}: {len(keys)} layers, shape {ft_weights[keys[0]].shape}, "
          f"top-5 avg singular values: {avg_sv[:5].round(4)}")
    return avg_sv


def plot_singular_values(avg_sv, module_name, output_dir, model_name, dataset):
    """Plot singular value decay and CEV for a single module type."""
    x = np.arange(1, len(avg_sv) + 1)
    energy = avg_sv ** 2
    cev = np.cumsum(energy) / np.sum(energy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: singular value decay
    ax1.semilogy(x, avg_sv, linewidth=1.5)
    ax1.set_xlabel("Singular value index", fontsize=12)
    ax1.set_ylabel("Singular value (log scale)", fontsize=12)
    ax1.set_title(f"Singular value decay — {module_name}", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Right: cumulative explained variance
    ax2.plot(x, cev, linewidth=1.5, color="tab:orange")
    # Mark LoRA rank reference lines
    for r in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if r < len(cev):
            ax2.axvline(r, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
            ax2.text(r, cev[min(r, len(cev)-1)] - 0.03, f"r={r}",
                     fontsize=7, ha="center", color="gray")
    ax2.axhline(0.95, color="red", linestyle=":", alpha=0.5, linewidth=1)
    ax2.text(len(cev) * 0.7, 0.955, "95%", fontsize=9, color="red", alpha=0.7)
    ax2.set_xlabel("Rank k", fontsize=12)
    ax2.set_ylabel("Cumulative explained variance", fontsize=12)
    ax2.set_title(f"CEV — {module_name}", fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"ΔW (Full FT − Base) — {model_name} / {dataset}", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"svd_{module_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_checkpoint_progression(all_checkpoint_svs, module_name, output_dir, model_name, dataset):
    """Plot SVD decay and CEV across multiple checkpoints for a single module type."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.viridis
    n = len(all_checkpoint_svs)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for (step, avg_sv), color in zip(all_checkpoint_svs, colors):
        x = np.arange(1, len(avg_sv) + 1)
        energy = avg_sv ** 2
        cev = np.cumsum(energy) / np.sum(energy)

        label = f"step {step}"
        ax1.semilogy(x, avg_sv, linewidth=1.2, color=color, label=label)
        ax2.plot(x, cev, linewidth=1.2, color=color, label=label)

    ax1.set_xlabel("Singular value index", fontsize=12)
    ax1.set_ylabel("Singular value (log scale)", fontsize=12)
    ax1.set_title(f"Singular value decay — {module_name}", fontsize=13)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for r in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if r < len(avg_sv):
            ax2.axvline(r, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
            ax2.text(r, 0.02, f"r={r}", fontsize=7, ha="center", color="gray")
    ax2.axhline(0.95, color="red", linestyle=":", alpha=0.5, linewidth=1)
    ax2.set_xlabel("Rank k", fontsize=12)
    ax2.set_ylabel("Cumulative explained variance", fontsize=12)
    ax2.set_title(f"CEV — {module_name}", fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"ΔW (Full FT − Base) across training — {model_name} / {dataset}", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"svd_progression_{module_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the base model directory (HF cache or local)")
    parser.add_argument("--ft_model_path", type=str, required=True,
                        help="Path to the full fine-tuned checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save plots")
    parser.add_argument("--model_name", type=str, default="",
                        help="Model name for plot titles")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name for plot titles")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading base model weights from: {args.base_model_path}")
    base_weights = load_safetensors_weights(args.base_model_path)
    print(f"  Loaded {len(base_weights)} tensors")

    # Find all checkpoints in the parent full FT directory
    ft_parent = os.path.dirname(args.ft_model_path)
    checkpoint_dirs = sorted(
        [d for d in os.listdir(ft_parent) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[1])
    )

    # Single checkpoint: original plots
    print(f"Loading fine-tuned weights from: {args.ft_model_path}")
    ft_weights = load_safetensors_weights(args.ft_model_path)
    print(f"  Loaded {len(ft_weights)} tensors")

    print("\nComputing SVD per module type (final checkpoint)...")
    for module_name in LINEAR_MODULES:
        avg_sv = compute_svd_for_module(ft_weights, base_weights, module_name)
        if avg_sv is not None:
            plot_singular_values(avg_sv, module_name, args.output_dir,
                                args.model_name, args.dataset)

    # Multi-checkpoint progression plots
    if len(checkpoint_dirs) >= 2:
        # Pick ~5 evenly spaced checkpoints including first and last
        if len(checkpoint_dirs) <= 5:
            selected = checkpoint_dirs
        else:
            indices = np.linspace(0, len(checkpoint_dirs) - 1, 5, dtype=int)
            selected = [checkpoint_dirs[i] for i in indices]

        print(f"\nCheckpoint progression: {[s for s in selected]}")

        # Collect SVDs per module across checkpoints
        module_checkpoint_svs = {m: [] for m in LINEAR_MODULES}

        for ckpt_name in selected:
            step = int(ckpt_name.split("-")[1])
            ckpt_path = os.path.join(ft_parent, ckpt_name)
            print(f"\n  Loading {ckpt_name}...")
            ckpt_weights = load_safetensors_weights(ckpt_path)

            for module_name in LINEAR_MODULES:
                avg_sv = compute_svd_for_module(ckpt_weights, base_weights, module_name)
                if avg_sv is not None:
                    module_checkpoint_svs[module_name].append((step, avg_sv))

            del ckpt_weights

        print("\nPlotting progression...")
        for module_name in LINEAR_MODULES:
            if module_checkpoint_svs[module_name]:
                plot_checkpoint_progression(
                    module_checkpoint_svs[module_name], module_name,
                    args.output_dir, args.model_name, args.dataset
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
