import os
from pathlib import Path
from typing import Iterable, List, Tuple
import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from main import LinearFineTuningExperiment


def _compute_single_experiment(args):

    i, rank_val, seed, dx, dy, n_train, noise_std, lora_ranks, base_seed, sv_decay_type, sv_decay_rate, sv_scale = args

    np.random.seed(base_seed + seed)
    exp = LinearFineTuningExperiment(dx, dy, rank_val, noise_std,
                                      sv_decay_type=sv_decay_type,
                                      sv_decay_rate=sv_decay_rate,
                                      sv_scale=sv_scale)

    # Generate dataset
    X_train, Y_train = exp.generate_dataset(n_train)

    # Solve FFT
    A_fft = exp.solve_fft(X_train, Y_train)
    fft_error = exp.compute_excess_risk(A_fft)

    # Solve LoRA for each rank
    lora_errors = {}
    for r in lora_ranks:
        A_lora = exp.solve_lora(X_train, Y_train, r)
        lora_errors[r] = exp.compute_excess_risk(A_lora)

    return i, seed, fft_error, lora_errors


def run_rank_sweep(
    true_rank_start: int = 1,
    true_rank_end: int = 40,
    true_rank_step: int = 1,
    dx: int = 50,
    dy: int = 50,
    n_train: int = 100,
    noise_std: float = 2,
    lora_ranks: Iterable[int] = (1, 2, 4, 8, 16, 32),
    n_seeds: int = 10,
    base_seed: int = 0,
    output_dir: str = "figures",
    use_log_x: bool = False,
    use_log_y: bool = True,
    sv_decay_type: str = None,
    sv_decay_rate: float = None,
    sv_scale: float = 1.0,
) -> Tuple[List[int], dict]:

    rank_values = list(range(true_rank_start, true_rank_end + 1, true_rank_step))
    if not rank_values:
        rank_values = [true_rank_start]

    # Set default decay rates if sv_decay_type is specified but rate is None
    if sv_decay_type is not None and sv_decay_rate is None:
        if sv_decay_type == "fast_decay":
            sv_decay_rate = 0.5
        elif sv_decay_type == "slow_decay":
            sv_decay_rate = 0.2

    print(f"Sweeping true_rank from {true_rank_start} to {true_rank_end} with step {true_rank_step}")
    print(f"Testing ranks: {rank_values}")
    print(f"Fixed dimensions: dx={dx}, dy={dy}")
    print(f"LoRA ranks: {list(lora_ranks)}")
    if sv_decay_type is not None:
        print(f"SV decay type: {sv_decay_type}, rate: {sv_decay_rate}, scale: {sv_scale}")

    # Prepare arguments for parallel computation
    args_list = []
    for i, rank_val in enumerate(rank_values):
        for seed in range(n_seeds):
            args_list.append((
                i, rank_val, seed, dx, dy, n_train, noise_std,
                lora_ranks, base_seed, sv_decay_type, sv_decay_rate, sv_scale
            ))

    n_processes = min(mp.cpu_count(), 8)  # Limit to 8
    print(f"Running {len(args_list)} experiments using {n_processes} processes...")

    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(_compute_single_experiment, args_list),
            total=len(args_list),
            desc="Computing experiments"
        ))

    fft_errors_by_rank = {rank: [] for rank in rank_values}
    lora_errors_by_rank = {rank: {r: [] for r in lora_ranks} for rank in rank_values}

    for i, seed, fft_error, lora_errors in results:
        rank_val = rank_values[i]
        fft_errors_by_rank[rank_val].append(fft_error)
        for r in lora_ranks:
            lora_errors_by_rank[rank_val][r].append(lora_errors[r])

    fft_stats = {
        "mean": [np.mean(fft_errors_by_rank[rank]) for rank in rank_values],
        "std": [np.std(fft_errors_by_rank[rank]) for rank in rank_values]
    }

    lora_stats = {}
    for r in lora_ranks:
        lora_stats[r] = {
            "mean": [np.mean(lora_errors_by_rank[rank][r]) for rank in rank_values],
            "std": [np.std(lora_errors_by_rank[rank][r]) for rank in rank_values]
        }

    print("Experiment completed. Generating plot...")

    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'  

    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'  
    plt.rcParams['axes.titlecolor'] = 'black'

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    fig.patch.set_facecolor('white') 
    ax.patch.set_facecolor('white')  
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']  

    ax.plot(
        rank_values,
        fft_stats["mean"],
        color="black",
        marker=markers[0],
        markersize=3,
        label="Full fine-tuning (FFT)",
    )
    ax.fill_between(
        rank_values,
        np.array(fft_stats["mean"]) - np.array(fft_stats["std"]),
        np.array(fft_stats["mean"]) + np.array(fft_stats["std"]),
        color="black",
        alpha=0.1,
    )

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lora_ranks)))
    for i, (color, r) in enumerate(zip(colors, lora_ranks)):
        means = lora_stats[r]["mean"]
        stds = lora_stats[r]["std"]
        marker = markers[(i + 1) % len(markers)]  # Start from index 1 for LoRA curves
        ax.plot(rank_values, means, marker=marker, markersize=5, color=color, label=f"LoRA (r={r})")
        ax.fill_between(
            rank_values,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=color,
            alpha=0.12,
        )

    ax.set_xlabel("True model rank")
    ax.set_ylabel("Excess Risk")

    if use_log_x:
        ax.set_xscale("log")
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())

    if use_log_y:
        ax.set_yscale("log")
        import matplotlib.ticker as ticker
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    if use_log_x and use_log_y:
        scale_suffix = "_loglog"
    elif use_log_x:
        scale_suffix = "_logx"
    elif use_log_y:
        scale_suffix = "_logy"
    else:
        scale_suffix = "_linear"

    # Add SV decay type to suffix
    sv_suffix = ""
    if sv_decay_type is not None:
        sv_suffix = f"_{sv_decay_type}"

    ax.set_title("LoRA vs. FFT across true model ranks")
    ax.legend(frameon=True, labelcolor='black')
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"rank_sweep{scale_suffix}{sv_suffix}.png"
    pdf_path = out_dir / f"rank_sweep{scale_suffix}{sv_suffix}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved figure to {png_path} and {pdf_path}")
    return rank_values, {"fft": fft_stats, "lora": lora_stats}


if __name__ == "__main__":
    # # Example 1: Default (no SV control) - current behavior
    print("\n" + "="*60)
    print("Running experiment 1: Default (no SV control)")
    print("="*60)
    run_rank_sweep(
        true_rank_start=1,
        true_rank_end=40,
        true_rank_step=1,
        dx=40,
        dy=40,
        n_train=30,
        noise_std=2,
        lora_ranks=(1, 2, 4, 8, 32),
        n_seeds=100,
        base_seed=0,
        output_dir="figures",
        use_log_x=False,
        use_log_y=True,
        sv_decay_type=None
    )

    # # Example 2: Constant singular values
    # print("\n" + "="*60)
    # print("Running experiment 2: Constant singular values")
    # print("="*60)
    # run_rank_sweep(
    #     true_rank_start=1,
    #     true_rank_end=40,
    #     true_rank_step=1,
    #     dx=40,
    #     dy=40,
    #     n_train=30,
    #     noise_std=2,
    #     lora_ranks=(1, 2, 4, 8, 32),
    #     n_seeds=100,
    #     base_seed=0,
    #     output_dir="figures",
    #     use_log_x=False,
    #     use_log_y=True,
    #     sv_decay_type="constant",
    #     sv_scale=8.0
    # )

    # # Example 3: Fast decay (exponential)
    # print("\n" + "="*60)
    # print("Running experiment 3: Fast decay (exponential)")
    # print("="*60)
    # run_rank_sweep(
    #     true_rank_start=1,
    #     true_rank_end=40,
    #     true_rank_step=1,
    #     dx=40,
    #     dy=40,
    #     n_train=30,
    #     noise_std=2,
    #     lora_ranks=(1, 2, 4, 8, 32),
    #     n_seeds=100,
    #     base_seed=0,
    #     output_dir="figures",
    #     use_log_x=False,
    #     use_log_y=True,
    #     sv_decay_type="fast_decay",
    #     sv_decay_rate=0.5,
    #     sv_scale=1.0
    # )

    # # Example 4: Slow decay (power-law)
    # print("\n" + "="*60)
    # print("Running experiment 4: Slow decay (power-law)")
    # print("="*60)
    # run_rank_sweep(
    #     true_rank_start=1,
    #     true_rank_end=40,
    #     true_rank_step=1,
    #     dx=40,
    #     dy=40,
    #     n_train=30,
    #     noise_std=2,
    #     lora_ranks=(1, 2, 4, 8, 32),
    #     n_seeds=100,
    #     base_seed=0,
    #     output_dir="figures",
    #     use_log_x=False,
    #     use_log_y=True,
    #     sv_decay_type="slow_decay",
    #     sv_decay_rate=0.2,
    #     sv_scale=1.0
    # )