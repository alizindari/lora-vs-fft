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
    Worker function for parallel computation of FFT and LoRA solutions for a single (n_train, seed) pair.

    Args:
        args: Tuple containing (i, n_train_val, seed, fixed_dx, fixed_dy, noise_std,
                               true_rank, lora_ranks, base_seed)

    Returns:
        Tuple of (i, seed, fft_error, lora_errors_dict)
    """
    i, n_train_val, seed, fixed_dx, fixed_dy, noise_std, true_rank, lora_ranks, base_seed = args

    # Dimensions are fixed
    dx = fixed_dx
    dy = fixed_dy

    # Set seed for reproducibility
    np.random.seed(base_seed + seed)

    # Initialize experiment
    exp = LinearFineTuningExperiment(dx, dy, true_rank, noise_std)

    # Generate dataset
    X_train, Y_train = exp.generate_dataset(n_train_val)

    # Compute FFT solution
    A_fft = exp.solve_fft(X_train, Y_train)
    fft_error = exp.compute_excess_risk(A_fft)

    # Compute LoRA solutions for all ranks
    lora_errors = {}
    for r in lora_ranks:
        A_lora = exp.solve_lora(X_train, Y_train, r)
        lora_errors[r] = exp.compute_excess_risk(A_lora)

    return i, seed, fft_error, lora_errors


def run_dimension_sweep(
    fixed_dx: int = 50,
    fixed_dy: int = 50,
    n_train_start: int = 10,
    n_train_end: int = 1000,
    n_train_multiplier: float = 2.0,  # How much to multiply n_train by each step
    noise_std: float = 12,
    true_rank: int = 1,
    lora_ranks: Iterable[int] = (1, 2, 4, 8),
    n_seeds: int = 5,
    base_seed: int = 0,
    output_dir: str = "figures",
    use_log_x: bool = True,
    use_log_y: bool = True,
) -> Tuple[List[float], dict]:
    """
    Sweep over training sample sizes while keeping dimensions fixed, and compare FFT vs. LoRA at
    multiple adapter ranks. Plots dimension-to-sample ratio on x-axis.
    Uses closed-form excess risk computation. Saves a high-quality plot (PNG + PDF).

    Args:
        fixed_dx: Fixed input dimension dx for all experiments.
        fixed_dy: Fixed output dimension dy for all experiments.
        n_train_start, n_train_end: Range for number of training samples.
        n_train_multiplier: Factor to multiply n_train by each step (default 2.0 = doubling).
        use_log_x: If True, use logarithmic scale for x-axis (dimension-to-sample ratio).
        use_log_y: If True, use logarithmic scale for y-axis (excess risk).
    """
    # Generate n_train values by multiplying from start to end
    n_train_values = []
    current_n = n_train_start
    while current_n <= n_train_end:
        n_train_values.append(int(current_n))
        current_n *= n_train_multiplier
    if not n_train_values:
        n_train_values = [n_train_start]

    # Reverse the order so that we go from small d/n to large d/n (left to right)
    # This means starting with large n (small d/n) and going to small n (large d/n)
    n_train_values.reverse()

    # Calculate dimension-to-sample ratios for each n_train value
    dim_ratios = [fixed_dx / n for n in n_train_values]

    # Prepare bookkeeping
    fft_stats = {"mean": [], "std": []}
    lora_stats = {r: {"mean": [], "std": []} for r in lora_ranks}

    # Use all available CPU cores
    n_cpus = mp.cpu_count()
    print(f"Using {n_cpus} CPU cores for parallel processing...")

    # Collect all (n_train, seed) combinations for parallel processing
    tasks = []
    for i, n_train_val in enumerate(n_train_values):
        for seed in range(n_seeds):
            task_args = (i, n_train_val, seed, fixed_dx, fixed_dy, noise_std,
                        true_rank, lora_ranks, base_seed)
            tasks.append(task_args)

    # Execute tasks in parallel with progress bar
    with mp.Pool(processes=n_cpus) as pool:
        results = list(tqdm(
            pool.imap(_compute_single_experiment, tasks),
            total=len(tasks),
            desc="Computing experiments",
            unit="task"
        ))

    # Organize results by n_train value
    results_by_n = {i: {"fft": [], "lora": {r: [] for r in lora_ranks}} for i in range(len(n_train_values))}

    for i, seed, fft_error, lora_errors in results:
        results_by_n[i]["fft"].append(fft_error)
        for r in lora_ranks:
            results_by_n[i]["lora"][r].append(lora_errors[r])

    # Aggregate results across seeds for each n_train value
    for i in range(len(n_train_values)):
        fft_errors = results_by_n[i]["fft"]
        fft_stats["mean"].append(float(np.mean(fft_errors)))
        fft_stats["std"].append(float(np.std(fft_errors)))

        for r in lora_ranks:
            lora_errors_r = results_by_n[i]["lora"][r]
            lora_stats[r]["mean"].append(float(np.mean(lora_errors_r)))
            lora_stats[r]["std"].append(float(np.std(lora_errors_r)))

    # Plotting
    plt.style.use("seaborn-v0_8-whitegrid")

    # Set serif font (Times-like) for all text and math
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX fonts for math which are Times-like

    # Force all text to be completely black
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'  # Black frame around axes
    plt.rcParams['axes.titlecolor'] = 'black'

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    # Set figure and axes frame to completely black
    fig.patch.set_facecolor('white')  # Figure background
    ax.patch.set_facecolor('white')   # Axes background
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    # Define markers for different curves
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']  # circle, square, triangle, diamond, etc.

    # FFT curve
    ax.plot(
        dim_ratios,
        fft_stats["mean"],
        color="black",
        marker=markers[0],
        label="Full fine-tuning (FFT)",
    )
    ax.fill_between(
        dim_ratios,
        np.array(fft_stats["mean"]) - np.array(fft_stats["std"]),
        np.array(fft_stats["mean"]) + np.array(fft_stats["std"]),
        color="black",
        alpha=0.1,
    )

    # LoRA curves
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lora_ranks)))
    for i, (color, r) in enumerate(zip(colors, lora_ranks)):
        means = lora_stats[r]["mean"]
        stds = lora_stats[r]["std"]
        marker = markers[(i + 1) % len(markers)]  # Start from index 1 for LoRA curves
        ax.plot(dim_ratios, means, marker=marker, color=color, label=f"LoRA (r={r})")
        ax.fill_between(
            dim_ratios,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=color,
            alpha=0.12,
        )

    # Always use dx/n for the x-axis (input dimension to sample ratio)
    xlabel = r"Input dimension-to-sample ratio $\frac{d_x}{n}$"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Excess Risk")

    # Add vertical line at ratio = 1
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label=r'$\frac{d_{\mathrm{x}}}{n} = 1$')

    # --- Independent axis scaling ---
    if use_log_x:
        ax.set_xscale("log")
        # Set x-axis ticks for log scale
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())

    if use_log_y:
        ax.set_yscale("log")
        # Use powers of 10 for Y-axis like X-axis
        import matplotlib.ticker as ticker
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    # Create suffix based on log scaling
    if use_log_x and use_log_y:
        scale_suffix = "_loglog"
    elif use_log_x:
        scale_suffix = "_logx"
    elif use_log_y:
        scale_suffix = "_logy"
    else:
        scale_suffix = "_linear"

    ax.legend(frameon=True, labelcolor='black')
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()

    # Save outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"dimension_sweep{scale_suffix}.png"
    pdf_path = out_dir / f"dimension_sweep{scale_suffix}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved figure to {png_path} and {pdf_path}")
    return dim_ratios, {"fft": fft_stats, "lora": lora_stats}


if __name__ == "__main__":
    # Example usage: sweep n_train while keeping dimensions fixed
    run_dimension_sweep(
        fixed_dx = 100,
        fixed_dy = 100,
        n_train_start = 10,
        n_train_end = 1000,
        n_train_multiplier = 1.2,  # Double n_train each step
        noise_std = 1,
        true_rank = 4,
        lora_ranks = (1, 2, 4, 8),
        n_seeds = 100,
        base_seed = 0,
        output_dir = "figures",
        use_log_x = True,
        use_log_y = True)

    # Example usage: sweep dy while keeping dx fixed
    # run_dimension_sweep(dx_start=50, dx_end=50, dy_start=5, dy_end=1000, dimension_multiplier=2.0, use_log_x=True, use_log_y=True)
    # run_dimension_sweep(dimension_multiplier=2.0, use_log_x=True, use_log_y=True)

