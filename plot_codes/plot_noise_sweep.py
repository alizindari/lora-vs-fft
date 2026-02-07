import os
from pathlib import Path
from typing import Iterable, List, Tuple
import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")  # Headless-friendly backend
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from main import LinearFineTuningExperiment


def _compute_single_experiment(args):
    """
    Worker function for parallel computation of FFT and LoRA solutions for a single (noise_level, seed) pair.

    Args:
        args: Tuple containing (i, noise_val, seed, dx, dy, n_train, true_rank, lora_ranks, base_seed)

    Returns:
        Tuple of (i, seed, fft_error, lora_errors_dict)
    """
    i, noise_val, seed, dx, dy, n_train, true_rank, lora_ranks, base_seed = args

    # Set seed for reproducibility
    np.random.seed(base_seed + seed)

    # Initialize experiment with the current noise level
    exp = LinearFineTuningExperiment(dx, dy, true_rank, noise_val)

    # Generate dataset
    X_train, Y_train = exp.generate_dataset(n_train)

    # Compute FFT solution
    A_fft = exp.solve_fft(X_train, Y_train)
    fft_error = exp.compute_excess_risk(A_fft)

    # Compute LoRA solutions for all ranks
    lora_errors = {}
    for r in lora_ranks:
        A_lora = exp.solve_lora(X_train, Y_train, r)
        lora_errors[r] = exp.compute_excess_risk(A_lora)

    return i, seed, fft_error, lora_errors


def run_noise_sweep(
    sigma_min: float = 0.1,
    sigma_max: float = 25.6,
    n_sigmas: int = 10,
    dx: int = 50,
    dy: int = 50,
    n_train: int = 100,
    true_rank: int = 1,
    lora_ranks: Iterable[int] = (1, 2, 4, 8),
    n_seeds: int = 5,
    base_seed: int = 0,
    output_dir: str = "figures",
    use_log_scale: bool = True,
) -> Tuple[List[float], dict]:
    """
    Sweep over noise levels (sigma) using exponential spacing (powers of 2) and compare FFT vs. LoRA at
    multiple adapter ranks. Plots noise level on x-axis with log scale.
    Uses closed-form excess risk computation. Saves a high-quality plot (PNG + PDF).

    Args:
        sigma_min, sigma_max: Range for noise standard deviation sigma. Uses powers of 2 spacing.
        n_sigmas: Number of sigma points to evaluate.
        dx, dy: Fixed input and output dimensions.
        n_train: Number of training samples (n).
        use_log_scale: If True, use logarithmic scale for both axes.
    """
    # Generate noise levels as powers of 2
    sigma_exponents = np.linspace(np.log2(sigma_min), np.log2(sigma_max), n_sigmas)
    sigma_values = 2 ** sigma_exponents
    sigma_values = list(np.unique(sigma_values))

    print(f"Sweeping noise levels: {sigma_values}")

    # Prepare bookkeeping
    fft_stats = {"mean": [], "std": []}
    lora_stats = {r: {"mean": [], "std": []} for r in lora_ranks}

    # Use all available CPU cores
    n_cpus = mp.cpu_count()
    print(f"Using {n_cpus} CPU cores for parallel processing...")

    # Collect all (noise_level, seed) combinations for parallel processing
    tasks = []
    for i, sigma_val in enumerate(sigma_values):
        for seed in range(n_seeds):
            task_args = (i, sigma_val, seed, dx, dy, n_train, true_rank, lora_ranks, base_seed)
            tasks.append(task_args)

    # Execute tasks in parallel with progress bar
    with mp.Pool(processes=n_cpus) as pool:
        results = list(tqdm(
            pool.imap(_compute_single_experiment, tasks),
            total=len(tasks),
            desc="Computing experiments",
            unit="task"
        ))

    # Organize results by noise level
    noise_results = {i: {"fft": [], "lora": {r: [] for r in lora_ranks}} for i in range(len(sigma_values))}

    for i, seed, fft_error, lora_errors in results:
        noise_results[i]["fft"].append(fft_error)
        for r in lora_ranks:
            noise_results[i]["lora"][r].append(lora_errors[r])

    # Aggregate results across seeds for each noise level
    for i in range(len(sigma_values)):
        fft_errors = noise_results[i]["fft"]
        fft_stats["mean"].append(float(np.mean(fft_errors)))
        fft_stats["std"].append(float(np.std(fft_errors)))

        for r in lora_ranks:
            lora_errors_r = noise_results[i]["lora"][r]
            lora_stats[r]["mean"].append(float(np.mean(lora_errors_r)))
            lora_stats[r]["std"].append(float(np.std(lora_errors_r)))

    # Plotting - DO NOT use seaborn style as it can override scale settings
    # plt.style.use("seaborn-v0_8-whitegrid")  # COMMENTED OUT - this was the problem!

    # Set font preferences manually instead
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['grid.alpha'] = 0.7

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    # Define markers for different curves
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']

    # FFT curve
    ax.plot(
        sigma_values,
        fft_stats["mean"],
        color="black",
        marker=markers[0],
        markersize=2,
        label="Full fine-tuning (FFT)",
    )
    fft_lower = np.maximum(np.array(fft_stats["mean"]) - np.array(fft_stats["std"]), 1e-10)
    ax.fill_between(
        sigma_values,
        fft_lower,
        np.array(fft_stats["mean"]) + np.array(fft_stats["std"]),
        color="black",
        alpha=0.1,
    )

    # LoRA curves
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lora_ranks)))
    for i, (color, r) in enumerate(zip(colors, lora_ranks)):
        means = lora_stats[r]["mean"]
        stds = lora_stats[r]["std"]
        marker = markers[(i + 1) % len(markers)]
        ax.plot(sigma_values, means, marker=marker, markersize=5, color=color, label=f"LoRA (r={r})")
        lora_lower = np.maximum(np.array(means) - np.array(stds), 1e-10)
        ax.fill_between(
            sigma_values,
            lora_lower,
            np.array(means) + np.array(stds),
            color=color,
            alpha=0.12,
        )

    # NOW set the scales AFTER all plotting and AFTER style settings
    ax.set_xscale("log")
    if use_log_scale:
        ax.set_yscale("log")
        scale_suffix = "_log"
    else:
        scale_suffix = ""

    # Labels and formatting
    ax.set_xlabel(r"Noise standard deviation $\sigma$")
    ax.set_ylabel("Excess Risk")
    ax.legend(frameon=True)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    
    fig.tight_layout()

    # Save outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"noise_sweep{scale_suffix}.png"
    pdf_path = out_dir / f"noise_sweep{scale_suffix}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved figure to {png_path} and {pdf_path}")
    print(f"Y-axis scale: {'log' if use_log_scale else 'linear'}")
    return sigma_values, {"fft": fft_stats, "lora": lora_stats}


if __name__ == "__main__":
    # Example usage: sweep noise levels with fixed dimensions
    run_noise_sweep(
        sigma_min=1,
        sigma_max=100,
        n_sigmas=40,
        dx=100,
        dy=100,
        n_train=1000,
        true_rank=10,
        lora_ranks=(1, 2, 4, 8, 16),
        n_seeds=100,
        base_seed=0,
        output_dir="figures",
        use_log_scale=True
    )