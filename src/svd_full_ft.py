"""
Compute layerwise SVD of the weight delta (W_ft - W_base) for a single fine-tuning checkpoint.
For each linear layer type (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj),
report normalized layerwise spectral summaries and aggregate them across transformer blocks.
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
from safetensors import safe_open


LINEAR_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ENERGY_THRESHOLDS = [0.80, 0.90, 0.95]
RANK_MARKERS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


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
    """Compute per-layer SVD of (W_ft - W_base), return list of singular value arrays."""
    keys = get_layer_keys(ft_weights, module_name)
    if not keys:
        print(f"  Warning: no keys found for {module_name}")
        return None

    all_singular_values = []
    for key in keys:
        delta = ft_weights[key] - base_weights[key]
        sv = np.linalg.svd(delta, compute_uv=False)
        all_singular_values.append(sv)

    print(f"  {module_name}: {len(keys)} layers, shape {ft_weights[keys[0]].shape}")
    return all_singular_values


def summarize_layerwise_svs(all_svs):
    """Normalize per-layer spectra and aggregate comparable layerwise summaries."""
    lengths = {len(sv) for sv in all_svs}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent singular value lengths across layers: {sorted(lengths)}")

    max_len = lengths.pop()
    normalized_svs = []
    cumulative_energy = []
    rank_stats = {threshold: [] for threshold in ENERGY_THRESHOLDS}

    for sv in all_svs:
        top_sv = sv[0] if len(sv) > 0 else 0.0
        if top_sv > 0:
            normalized_svs.append(sv / top_sv)
        else:
            normalized_svs.append(np.zeros_like(sv))

        energy = sv ** 2
        total = np.sum(energy)
        if total > 0:
            cvr = np.cumsum(energy) / total
        else:
            cvr = np.ones(len(sv))
        cumulative_energy.append(cvr)

        for threshold in ENERGY_THRESHOLDS:
            rank = int(np.searchsorted(cvr, threshold, side="left") + 1)
            rank_stats[threshold].append(rank)

    normalized_svs = np.stack(normalized_svs)
    cumulative_energy = np.stack(cumulative_energy)

    return {
        "max_len": max_len,
        "median_norm_sv": np.median(normalized_svs, axis=0),
        "q25_norm_sv": np.percentile(normalized_svs, 25, axis=0),
        "q75_norm_sv": np.percentile(normalized_svs, 75, axis=0),
        "median_cvr": np.median(cumulative_energy, axis=0),
        "q25_cvr": np.percentile(cumulative_energy, 25, axis=0),
        "q75_cvr": np.percentile(cumulative_energy, 75, axis=0),
        "rank_stats": rank_stats,
    }


def print_rank_summary(module_name, summary):
    """Print concise rank-at-threshold stats for one module type."""
    parts = []
    for threshold in ENERGY_THRESHOLDS:
        ranks = np.array(summary["rank_stats"][threshold], dtype=float)
        parts.append(
            f"rank@{int(threshold * 100)} mean={ranks.mean():.1f}, median={np.median(ranks):.1f}"
        )
    print(f"    {module_name} summary: " + " | ".join(parts))


def plot_singular_values(summary, module_name, output_dir, model_name, dataset, checkpoint):
    """Plot normalized spectral decay and layerwise cumulative variance summaries."""
    x = np.arange(1, summary["max_len"] + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogy(x, summary["median_norm_sv"], linewidth=1.8, label="Median normalized SV")
    ax1.fill_between(
        x,
        summary["q25_norm_sv"],
        summary["q75_norm_sv"],
        alpha=0.2,
        label="25-75 percentile",
    )
    ax1.set_xlabel("Singular value index", fontsize=12)
    ax1.set_ylabel("Normalized singular value (log scale)", fontsize=12)
    ax1.set_title(f"Layerwise normalized decay — {module_name}", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(x, summary["median_cvr"], linewidth=1.8, color="tab:orange", label="Median CVR")
    ax2.fill_between(
        x,
        summary["q25_cvr"],
        summary["q75_cvr"],
        color="tab:orange",
        alpha=0.2,
        label="25-75 percentile",
    )
    for r in RANK_MARKERS:
        if r < len(summary["median_cvr"]):
            ax2.axvline(r, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    for thresh in ENERGY_THRESHOLDS:
        ax2.axhline(thresh, color="red", linestyle=":", alpha=0.5, linewidth=1)
        ax2.text(
            summary["max_len"] * 0.7,
            thresh + 0.005,
            f"{int(thresh * 100)}%",
            fontsize=9,
            color="red",
            alpha=0.7,
        )
    ax2.set_xlabel("Rank k", fontsize=12)
    ax2.set_ylabel("Cumulative variance ratio", fontsize=12)
    ax2.set_title(f"Layerwise cumulative variance — {module_name}", fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle(f"ΔW (Full FT − Base) — {model_name} / {dataset} / {checkpoint}", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"svd_{module_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the base model directory (HF cache or local)")
    parser.add_argument("--ft_checkpoint_path", type=str, required=True,
                        help="Path to a single fine-tuned checkpoint directory")
    parser.add_argument("--model_name", type=str, default="",
                        help="Model name for plot titles")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name for plot titles")
    args = parser.parse_args()

    checkpoint_name = os.path.basename(args.ft_checkpoint_path)
    run_dir = os.path.basename(os.path.dirname(args.ft_checkpoint_path))
    parts = args.ft_checkpoint_path.rstrip("/").split("/")
    try:
        idx = parts.index("new_saves")
        rel_path = os.path.join(*parts[idx+1:])
    except ValueError:
        rel_path = os.path.join(run_dir, checkpoint_name)

    output_dir = os.path.join("plots", rel_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base model weights from: {args.base_model_path}")
    base_weights = load_safetensors_weights(args.base_model_path)
    print(f"  Loaded {len(base_weights)} tensors")

    print(f"Loading fine-tuned weights from: {args.ft_checkpoint_path}")
    ft_weights = load_safetensors_weights(args.ft_checkpoint_path)
    print(f"  Loaded {len(ft_weights)} tensors")

    print("\nComputing SVD per module type...")
    for module_name in LINEAR_MODULES:
        all_svs = compute_svd_for_module(ft_weights, base_weights, module_name)
        if all_svs is not None:
            summary = summarize_layerwise_svs(all_svs)
            print_rank_summary(module_name, summary)
            plot_singular_values(summary, module_name, output_dir,
                                 args.model_name, args.dataset, checkpoint_name)

    print(f"\nAll plots saved to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
