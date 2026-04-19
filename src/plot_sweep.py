"""
Plot accuracy vs. sample size n for a single benchmark, one curve per
fine-tuning method (FFT and LoRA at multiple ranks), mean +/- std over
seeds. Mirrors the style of Fig. 1(a) in the paper.

Auto-discovers all eval_results.json files under
    <saves_root>/<dataset>_<n>/<method>/<seed_dir>/checkpoint-*/eval_results.json
where <method> is `full` or `lora_r<R>_a<A>`.

Usage:
  python src/plot_sweep.py \\
      --saves_root saves/qwen2.5-0.5b \\
      --dataset boolq_alpaca \\
      --task boolq \\
      --output figures/boolq_sample_size_sweep.pdf
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# Times-like typography (STIXGeneral is available system-wide and designed
# to match Times; falls back to any installed Times face if present).
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "axes.labelsize":    13,
    "axes.titlesize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "axes.linewidth":    0.9,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.minor.size":  2.5,
    "ytick.minor.size":  2.5,
})


METHOD_ORDER = [
    "full",
    "lora_r1_a2",
    "lora_r8_a16",
    "lora_r16_a32",
    "lora_r32_a64",
    "lora_r64_a128",
    "lora_r128_a256",
    "lora_r256_a512",
]
METHOD_LABEL = {
    "full": "Full FT",
    "lora_r1_a2": "LoRA r=1",
    "lora_r8_a16": "LoRA r=8",
    "lora_r16_a32": "LoRA r=16",
    "lora_r32_a64": "LoRA r=32",
    "lora_r64_a128": "LoRA r=64",
    "lora_r128_a256": "LoRA r=128",
    "lora_r256_a512": "LoRA r=256",
}


def parse_results(saves_root, dataset, task):
    """Walk the save tree and collect accuracies grouped by (n, method)."""
    pattern = os.path.join(
        saves_root,
        f"{dataset}_*",
        "*",
        "*",
        "checkpoint-*",
        "eval_results.json",
    )
    files = glob.glob(pattern)

    data = defaultdict(lambda: defaultdict(list))  # data[n][method] = [acc, ...]
    for f in files:
        m = re.search(rf"{re.escape(dataset)}_(\d+)", f)
        if not m:
            continue
        n = int(m.group(1))
        m2 = re.search(r"/(full|lora_r\d+_a\d+)/", f)
        if not m2:
            continue
        method = m2.group(1)
        with open(f) as fh:
            results = json.load(fh).get("accuracies", {})
        if task not in results:
            continue
        data[n][method].append(results[task]["acc"] * 100)
    return data


def discover_methods(data):
    seen = set()
    for mdict in data.values():
        seen.update(mdict.keys())
    ordered = [m for m in METHOD_ORDER if m in seen]
    extras = sorted(seen - set(ordered))
    return ordered + extras


def assign_colors(methods):
    """FFT in black; LoRA ranks along plasma (low rank = dark purple -> high = orange).

    Plasma gives wider hue separation than viridis, so adjacent ranks stay
    distinguishable while still preserving rank ordering.
    """
    cmap = plt.get_cmap("plasma")
    colors = {}
    lora_methods = [m for m in methods if m.startswith("lora_")]
    n_lora = max(1, len(lora_methods))
    for i, m in enumerate(lora_methods):
        t = i / max(1, n_lora - 1)
        colors[m] = cmap(0.05 + 0.80 * t)
    if "full" in methods:
        colors["full"] = "#000000"
    return colors


def load_baseline(baseline_path, task):
    """Return zero-shot accuracy (%) for task from an eval_results.json, or None."""
    if not baseline_path or not os.path.isfile(baseline_path):
        return None
    with open(baseline_path) as fh:
        results = json.load(fh).get("accuracies", {})
    if task not in results or results[task].get("acc") is None:
        return None
    return results[task]["acc"] * 100


def make_plot(data, output_path, title=None, ylabel=None, task="",
              style="paper", baseline_acc=None):
    """Render the sweep figure.

    style:
      "paper"    : mirrors Fig. 1(a): y = error rate (log), x = n (log,
                   reversed so big-n is on the left like the paper's dx/n).
      "accuracy" : raw accuracy on linear y, n increasing left-to-right.
    """
    methods = discover_methods(data)
    if not methods:
        raise SystemExit("No methods found in data.")
    ns_all = sorted(data.keys())
    colors = assign_colors(methods)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for m in methods:
        xs, mu, sd = [], [], []
        for n in ns_all:
            accs = data[n].get(m, [])
            if not accs:
                continue
            xs.append(n)
            if style == "paper":
                # Convert accuracies (%) to error rates so lower = better.
                err = [100.0 - a for a in accs]
                mu.append(np.mean(err))
                sd.append(np.std(err, ddof=1) if len(err) > 1 else 0.0)
            else:
                mu.append(np.mean(accs))
                sd.append(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)
        if not xs:
            continue
        xs, mu, sd = np.array(xs), np.array(mu), np.array(sd)
        c = colors[m]
        lw = 2.1 if m == "full" else 1.7
        marker = "s" if m == "full" else "o"
        ax.fill_between(xs, mu - sd, mu + sd, color=c, alpha=0.15, linewidth=0)
        ax.plot(
            xs, mu,
            label=METHOD_LABEL.get(m, m),
            color=c,
            marker=marker,
            markersize=5.5,
            markeredgewidth=0,
            lw=lw,
        )

    if baseline_acc is not None:
        y_base = (100.0 - baseline_acc) if style == "paper" else baseline_acc
        ax.axhline(y_base, ls="--", lw=1.2, color="0.35",
                   label=r"Zero-shot $A_0$", zorder=1)

    ax.set_xscale("log")
    if style == "paper":
        ax.set_yscale("log")
        ax.invert_xaxis()  # big n on the left, small n on the right (like paper's dx/n)
        ax.set_xlabel(r"Number of training samples $n$")
        ax.set_ylabel(ylabel or f"{task.upper()} error rate (\\%)".replace("\\", ""))
    else:
        ax.set_xlabel(r"Number of training samples $n$")
        ax.set_ylabel(ylabel or f"{task.upper()} accuracy (\\%)".replace("\\", ""))

    if title:
        ax.set_title(title)
    ax.grid(True, which="major", axis="both", alpha=0.25, linewidth=0.5)
    ax.grid(True, which="minor", axis="both", alpha=0.12, linewidth=0.4)

    legend_loc = "upper left" if style == "paper" else "lower right"
    leg = ax.legend(frameon=False, loc=legend_loc, handlelength=2.2,
                    handletextpad=0.5, labelspacing=0.4, borderaxespad=0.6)
    for line in leg.get_lines():
        line.set_linewidth(1.8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base = os.path.splitext(output_path)[0]
    pdf, png = base + ".pdf", base + ".png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    print(f"saved: {pdf}\nsaved: {png}")


def print_summary(data):
    methods = discover_methods(data)
    ns = sorted(data.keys())
    n_cells = sum(len(data[n]) for n in ns)
    print(f"\nFound {n_cells} (n, method) cells across {len(ns)} sample sizes "
          f"and {len(methods)} methods.")
    print(f"\n{'n':<8}{'method':<22}{'seeds':<8}{'mean':<10}{'std':<8}")
    for n in ns:
        for m in methods:
            accs = data[n].get(m, [])
            if not accs:
                continue
            mu = np.mean(accs)
            sd = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
            print(f"{n:<8}{m:<22}{len(accs):<8}{mu:<10.2f}{sd:<8.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saves_root", default="saves/qwen2.5-0.5b")
    p.add_argument("--dataset", default="boolq_alpaca",
                   help="dataset short name (matches <dataset>_<n> dirs)")
    p.add_argument("--task", default="boolq",
                   help="lm-eval task name whose accuracy to plot")
    p.add_argument("--output", default=None,
                   help="output PDF path (PNG is also saved alongside)")
    p.add_argument("--title", default=None)
    p.add_argument("--ylabel", default=None)
    p.add_argument("--style", choices=["paper", "accuracy"], default="paper",
                   help="'paper' = error rate on log y, n reversed (mirrors Fig 1a); "
                        "'accuracy' = raw accuracy, n increasing left-to-right")
    p.add_argument("--baseline_path", default=None,
                   help="eval_results.json from the base (un-tuned) model. "
                        "If provided, draw a dashed horizontal line at its task accuracy.")
    args = p.parse_args()

    output = args.output or f"figures/{args.dataset}_{args.task}_sample_size_sweep.pdf"

    data = parse_results(args.saves_root, args.dataset, args.task)
    if not data:
        raise SystemExit(
            f"No eval results found under {args.saves_root}/{args.dataset}_*/.\n"
            f"Looked for: <saves_root>/<dataset>_<n>/<method>/<seed>/checkpoint-*/eval_results.json"
        )
    print_summary(data)
    baseline_acc = load_baseline(args.baseline_path, args.task)
    if args.baseline_path and baseline_acc is None:
        print(f"warning: baseline file {args.baseline_path} missing or has no "
              f"'{args.task}' entry; skipping baseline line.")
    make_plot(data, output, args.title, args.ylabel, args.task,
              style=args.style, baseline_acc=baseline_acc)


if __name__ == "__main__":
    main()
