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
    """
    Compute a single experiment for a given decay rate and seed.
    """
    i, decay_rate, seed, dx, dy, true_rank, n_train, noise_std, lora_ranks, base_seed, sv_scale = args

    np.random.seed(base_seed + seed)
    exp = LinearFineTuningExperiment(
        dx, dy, true_rank, noise_std,
        sv_decay_type="fast_decay",
        sv_decay_rate=decay_rate,
        sv_scale=sv_scale
    )

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


def plot_singular_values(
    decay_rates: List[float],
    true_rank: int,
    sv_scale: float = 1.0,
    output_dir: str = "figures",
):
    """
    Plot singular values for different decay rates.
    """
    print("\nGenerating singular value visualization...")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('white')

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(decay_rates)))

    sv_indices = np.arange(1, true_rank + 1)

    for decay_rate, color in zip(decay_rates, colors):
        # Generate singular values
        if decay_rate == 0:
            svs = sv_scale * np.ones(true_rank)
            label = f"Constant (rate=0)"
        else:
            svs = sv_scale * np.exp(-decay_rate * np.arange(true_rank))
            label = f"Decay rate={decay_rate:.2f}"

        ax.plot(sv_indices, svs, color=color, marker='o', markersize=4, label=label, linewidth=2)

    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value Magnitude")
    ax.set_title("Singular Value Profiles Across Decay Rates")
    ax.set_yscale("log")
    ax.legend(frameon=True, loc='best')
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "singular_values_decay_rates.png"
    pdf_path = out_dir / "singular_values_decay_rates.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved singular value plot to {png_path} and {pdf_path}")


def run_decay_rate_sweep(
    decay_rate_start: float = 0.0,
    decay_rate_end: float = 2.0,
    decay_rate_step: float = 0.2,
    dx: int = 50,
    dy: int = 50,
    n_train: int = 100,
    noise_std: float = 2.0,
    lora_ranks: Iterable[int] = (1, 2, 4, 8, 16, 32),
    n_seeds: int = 10,
    base_seed: int = 0,
    sv_scale: float = 1.0,
    output_dir: str = "figures",
) -> Tuple[List[float], dict]:
    """
    Run experiments sweeping over SV decay rates with full-rank Delta_star.

    Args:
        decay_rate_start: Starting decay rate (0 = constant SVs)
        decay_rate_end: Ending decay rate
        decay_rate_step: Step size for decay rate
        dx: Input dimension
        dy: Output dimension
        n_train: Number of training samples
        noise_std: Noise standard deviation
        lora_ranks: LoRA ranks to test
        n_seeds: Number of random seeds per configuration
        base_seed: Base random seed
        sv_scale: Scale factor for singular values
        output_dir: Output directory for plots

    Returns:
        decay_rates: List of decay rates tested
        stats: Dictionary of statistics
    """
    # Full rank Delta_star
    true_rank = min(dx, dy)

    # Generate decay rate values
    decay_rates = np.arange(decay_rate_start, decay_rate_end + decay_rate_step/2, decay_rate_step).tolist()

    print(f"Decay Rate Sweep Experiment")
    print(f"="*60)
    print(f"Sweeping decay rate from {decay_rate_start} to {decay_rate_end} with step {decay_rate_step}")
    print(f"Decay rates: {decay_rates}")
    print(f"Fixed dimensions: dx={dx}, dy={dy}")
    print(f"True rank (full): {true_rank}")
    print(f"LoRA ranks: {list(lora_ranks)}")
    print(f"Number of seeds: {n_seeds}")

    # First, plot the singular values
    plot_singular_values(decay_rates, true_rank, sv_scale, output_dir)

    # Prepare arguments for parallel computation
    args_list = []
    for i, decay_rate in enumerate(decay_rates):
        for seed in range(n_seeds):
            args_list.append((
                i, decay_rate, seed, dx, dy, true_rank, n_train, noise_std,
                lora_ranks, base_seed, sv_scale
            ))

    n_processes = min(mp.cpu_count(), 8)
    print(f"Running {len(args_list)} experiments using {n_processes} processes...")

    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(_compute_single_experiment, args_list),
            total=len(args_list),
            desc="Computing experiments"
        ))

    # Organize results
    fft_errors_by_rate = {rate: [] for rate in decay_rates}
    lora_errors_by_rate = {rate: {r: [] for r in lora_ranks} for rate in decay_rates}

    for i, seed, fft_error, lora_errors in results:
        decay_rate = decay_rates[i]
        fft_errors_by_rate[decay_rate].append(fft_error)
        for r in lora_ranks:
            lora_errors_by_rate[decay_rate][r].append(lora_errors[r])

    # Compute statistics
    fft_stats = {
        "mean": [np.mean(fft_errors_by_rate[rate]) for rate in decay_rates],
        "std": [np.std(fft_errors_by_rate[rate]) for rate in decay_rates]
    }

    lora_stats = {}
    for r in lora_ranks:
        lora_stats[r] = {
            "mean": [np.mean(lora_errors_by_rate[rate][r]) for rate in decay_rates],
            "std": [np.std(lora_errors_by_rate[rate][r]) for rate in decay_rates]
        }

    print("\nExperiment completed. Generating decay rate sweep plot...")

    # Plotting
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

    # Plot FFT
    ax.plot(
        decay_rates,
        fft_stats["mean"],
        color="black",
        marker=markers[0],
        markersize=5,
        linewidth=2,
        label="Full fine-tuning (FFT)",
    )
    ax.fill_between(
        decay_rates,
        np.array(fft_stats["mean"]) - np.array(fft_stats["std"]),
        np.array(fft_stats["mean"]) + np.array(fft_stats["std"]),
        color="black",
        alpha=0.1,
    )

    # Plot LoRA for different ranks
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lora_ranks)))
    for i, (color, r) in enumerate(zip(colors, lora_ranks)):
        means = lora_stats[r]["mean"]
        stds = lora_stats[r]["std"]
        marker = markers[(i + 1) % len(markers)]
        ax.plot(decay_rates, means, marker=marker, markersize=5, linewidth=2,
                color=color, label=f"LoRA (r={r})")
        ax.fill_between(
            decay_rates,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=color,
            alpha=0.12,
        )

    ax.set_xlabel("Singular Value Decay Rate")
    ax.set_ylabel("Excess Risk")
    ax.set_yscale("log")

    import matplotlib.ticker as ticker
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    # ax.set_title(f"LoRA vs FFT: Full-Rank Delta (rank={true_rank})")
    ax.legend(frameon=True, labelcolor='black', loc='best')
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "decay_rate_sweep.png"
    pdf_path = out_dir / "decay_rate_sweep.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved decay rate sweep plot to {png_path} and {pdf_path}")
    return decay_rates, {"fft": fft_stats, "lora": lora_stats}


if __name__ == "__main__":
    # Optimized hyperparameters for clear FFT→LoRA transition:
    # - Flat SVs (decay=0): All directions important, need full capacity → FFT wins
    # - Concentrated SVs (high decay): Few directions important → LoRA wins
    # - Key: Lots of clean data so FFT can learn accurately when all directions matter
    run_decay_rate_sweep(
        decay_rate_start=0.0,      # 0 = constant SVs (FFT should win)
        decay_rate_end=2.5,        # High decay = concentrated SVs (LoRA should win)
        decay_rate_step=0.125,
        dx=40,
        dy=40,
        n_train=200,               # Lots of data for accurate learning
        noise_std=0.5,             # Low noise so signal is clear
        lora_ranks=(1, 2, 4, 8, 16),
        n_seeds=100,
        base_seed=42,
        sv_scale=5.0,              # Strong signal
        output_dir="figures",
    )
