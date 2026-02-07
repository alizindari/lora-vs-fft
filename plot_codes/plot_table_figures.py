import os
from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")  # Headless-friendly backend
import matplotlib.pyplot as plt
import numpy as np





def parse_table_tex(file_path):
    """
    Parse the LaTeX table and extract accuracy data.
    
    Returns:
        dict: {dataset: {size: {method: accuracy}}}
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract data rows for each dataset
    datasets = {}
    
    # Pattern to match dataset rows
    # BoolQ rows
    boolq_pattern = r'BoolQ.*?0\.5B.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?1\.5B.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?3B.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?7B.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)'
    
    # Simpler approach: extract all numbers from each dataset section
    datasets_data = {}
    
    # Split by dataset
    boolq_section = re.search(r'BoolQ.*?ARC-Easy', content, re.DOTALL)
    arc_easy_section = re.search(r'ARC-Easy.*?ARC-Challenge', content, re.DOTALL)
    
    def extract_numbers_from_line(line):
        # Remove textcolor commands and extract numbers
        line = re.sub(r'\\textcolor\{blue\}\{', '', line)
        line = re.sub(r'\}', '', line)
        # Extract all decimal numbers
        numbers = re.findall(r'\d+\.\d+', line)
        return [float(n) for n in numbers]
    
    # Parse BoolQ
    if boolq_section:
        lines = boolq_section.group(0).split('&')
        boolq_lines = []
        for line in content.split('\n'):
            if 'BoolQ' in line or ('0.5B' in line and 'BoolQ' in content[content.find('BoolQ'):content.find('ARC-Easy')]):
                if '0.5B' in line or '1.5B' in line or '3B' in line or '7B' in line:
                    boolq_lines.append(line)
        
        datasets_data['BoolQ'] = {}
        for line in boolq_lines:
            if '0.5B' in line:
                nums = extract_numbers_from_line(line)
                if len(nums) >= 10:
                    datasets_data['BoolQ']['0.5B'] = {
                        'FFT': nums[0],
                        'LoRA_1': nums[1], 'LoRA_2': nums[2], 'LoRA_4': nums[3],
                        'LoRA_8': nums[4], 'LoRA_16': nums[5], 'LoRA_32': nums[6],
                        'LoRA_64': nums[7], 'LoRA_128': nums[8], 'LoRA_256': nums[9]
                    }
            elif '1.5B' in line:
                nums = extract_numbers_from_line(line)
                if len(nums) >= 10:
                    datasets_data['BoolQ']['1.5B'] = {
                        'FFT': nums[0],
                        'LoRA_1': nums[1], 'LoRA_2': nums[2], 'LoRA_4': nums[3],
                        'LoRA_8': nums[4], 'LoRA_16': nums[5], 'LoRA_32': nums[6],
                        'LoRA_64': nums[7], 'LoRA_128': nums[8], 'LoRA_256': nums[9]
                    }
            elif '3B' in line:
                nums = extract_numbers_from_line(line)
                if len(nums) >= 10:
                    datasets_data['BoolQ']['3B'] = {
                        'FFT': nums[0],
                        'LoRA_1': nums[1], 'LoRA_2': nums[2], 'LoRA_4': nums[3],
                        'LoRA_8': nums[4], 'LoRA_16': nums[5], 'LoRA_32': nums[6],
                        'LoRA_64': nums[7], 'LoRA_128': nums[8], 'LoRA_256': nums[9]
                    }
            elif '7B' in line:
                nums = extract_numbers_from_line(line)
                if len(nums) >= 10:
                    datasets_data['BoolQ']['7B'] = {
                        'FFT': nums[0],
                        'LoRA_1': nums[1], 'LoRA_2': nums[2], 'LoRA_4': nums[3],
                        'LoRA_8': nums[4], 'LoRA_16': nums[5], 'LoRA_32': nums[6],
                        'LoRA_64': nums[7], 'LoRA_128': nums[8], 'LoRA_256': nums[9]
                    }
    
    # Parse ARC-Easy
    if arc_easy_section:
        datasets_data['ARC-Easy'] = {}
        for line in content.split('\n'):
            if 'ARC-Easy' in line or ('0.5B' in line and 'ARC-Easy' in content[content.find('ARC-Easy'):content.find('ARC-Challenge')]):
                if '0.5B' in line or '1.5B' in line or '3B' in line or '7B' in line:
                    nums = extract_numbers_from_line(line)
                    if len(nums) >= 10:
                        if '0.5B' in line:
                            datasets_data['ARC-Easy']['0.5B'] = {
                                'FFT': nums[0],
                                'LoRA_1': nums[1], 'LoRA_2': nums[2], 'LoRA_4': nums[3],
                                'LoRA_8': nums[4], 'LoRA_16': nums[5], 'LoRA_32': nums[6],
                                'LoRA_64': nums[7], 'LoRA_128': nums[8], 'LoRA_256': nums[9]
                            }
                        elif '1.5B' in line:
                            datasets_data['ARC-Easy']['1.5B'] = {
                                'FFT': nums[0],
                                'LoRA_1': nums[1], 'LoRA_2': nums[2], 'LoRA_4': nums[3],
                                'LoRA_8': nums[4], 'LoRA_16': nums[5], 'LoRA_32': nums[6],
                                'LoRA_64': nums[7], 'LoRA_128': nums[8], 'LoRA_256': nums[9]
                            }
                        elif '3B' in line:
                            datasets_data['ARC-Easy']['3B'] = {
                                'FFT': nums[0],
                                'LoRA_1': nums[1], 'LoRA_2': nums[2], 'LoRA_4': nums[3],
                                'LoRA_8': nums[4], 'LoRA_16': nums[5], 'LoRA_32': nums[6],
                                'LoRA_64': nums[7], 'LoRA_128': nums[8], 'LoRA_256': nums[9]
                            }
                        elif '7B' in line:
                            datasets_data['ARC-Easy']['7B'] = {
                                'FFT': nums[0],
                                'LoRA_1': nums[1], 'LoRA_2': nums[2], 'LoRA_4': nums[3],
                                'LoRA_8': nums[4], 'LoRA_16': nums[5], 'LoRA_32': nums[6],
                                'LoRA_64': nums[7], 'LoRA_128': nums[8], 'LoRA_256': nums[9]
                            }
    
    return datasets_data


def parse_table_tex_manual(file_path):
    """
    Manually parse the table based on the known structure.
    """
    data = {
        'BoolQ': {
            '0.5B': {'FFT': 82.2, 'LoRA_1': 64.6, 'LoRA_2': 65.5, 'LoRA_4': 62.5, 'LoRA_8': 70.4, 
                     'LoRA_16': 75.4, 'LoRA_32': 75.6, 'LoRA_64': 80.7, 'LoRA_128': 75.4, 'LoRA_256': 80.0},
            '1.5B': {'FFT': 85.9, 'LoRA_1': 81.0, 'LoRA_2': 81.5, 'LoRA_4': 83.4, 'LoRA_8': 82.8,
                     'LoRA_16': 84.7, 'LoRA_32': 86.9, 'LoRA_64': 85.0, 'LoRA_128': 84.9, 'LoRA_256': 85.6},
            '3B': {'FFT': 88.5, 'LoRA_1': 86.9, 'LoRA_2': 87.1, 'LoRA_4': 87.4, 'LoRA_8': 87.4,
                   'LoRA_16': 87.5, 'LoRA_32': 87.3, 'LoRA_64': 88.3, 'LoRA_128': 86.8, 'LoRA_256': 88.0},
            '7B': {'FFT': 89.7, 'LoRA_1': 89.3, 'LoRA_2': 89.5, 'LoRA_4': 89.7, 'LoRA_8': 89.7,
                   'LoRA_16': 89.6, 'LoRA_32': 89.5, 'LoRA_64': 90.0, 'LoRA_128': 89.7, 'LoRA_256': 89.7}
        },
        'ARC-Easy': {
            '0.5B': {'FFT': 61.0, 'LoRA_1': 64.6, 'LoRA_2': 65.6, 'LoRA_4': 65.1, 'LoRA_8': 65.5,
                     'LoRA_16': 65.8, 'LoRA_32': 65.1, 'LoRA_64': 64.7, 'LoRA_128': 64.5, 'LoRA_256': 64.4},
            '1.5B': {'FFT': 73.7, 'LoRA_1': 76.8, 'LoRA_2': 77.5, 'LoRA_4': 79.0, 'LoRA_8': 79.1,
                     'LoRA_16': 79.1, 'LoRA_32': 78.7, 'LoRA_64': 79.1, 'LoRA_128': 80.0, 'LoRA_256': 80.7},
            '3B': {'FFT': 80.9, 'LoRA_1': 78.2, 'LoRA_2': 79.0, 'LoRA_4': 81.0, 'LoRA_8': 79.6,
                   'LoRA_16': 80.2, 'LoRA_32': 80.5, 'LoRA_64': 80.5, 'LoRA_128': 79.5, 'LoRA_256': 81.9},
            '7B': {'FFT': 85.0, 'LoRA_1': 82.4, 'LoRA_2': 83.3, 'LoRA_4': 83.2, 'LoRA_8': 82.9,
                   'LoRA_16': 82.7, 'LoRA_32': 82.8, 'LoRA_64': 82.6, 'LoRA_128': 82.8, 'LoRA_256': 82.2}
        },
        'ARC-Challenge': {
            '0.5B': {'FFT': 30.9, 'LoRA_1': 29.6, 'LoRA_2': 30.0, 'LoRA_4': 30.2, 'LoRA_8': 31.4,
                     'LoRA_16': 30.4, 'LoRA_32': 31.4, 'LoRA_64': 31.4, 'LoRA_128': 30.6, 'LoRA_256': 31.1},
            '1.5B': {'FFT': 44.8, 'LoRA_1': 40.9, 'LoRA_2': 40.3, 'LoRA_4': 41.7, 'LoRA_8': 41.1,
                     'LoRA_16': 41.8, 'LoRA_32': 40.8, 'LoRA_64': 42.2, 'LoRA_128': 41.4, 'LoRA_256': 42.7},
            '3B': {'FFT': 47.3, 'LoRA_1': 44.6, 'LoRA_2': 44.2, 'LoRA_4': 44.1, 'LoRA_8': 45.0,
                   'LoRA_16': 45.2, 'LoRA_32': 48.2, 'LoRA_64': 48.8, 'LoRA_128': 49.0, 'LoRA_256': 48.2},
            '7B': {'FFT': 53.5, 'LoRA_1': 49.8, 'LoRA_2': 53.1, 'LoRA_4': 52.9, 'LoRA_8': 53.1,
                   'LoRA_16': 52.3, 'LoRA_32': 52.9, 'LoRA_64': 55.8, 'LoRA_128': 54.7, 'LoRA_256': 55.7}
        },
        'Commonsense-QA': {
            '0.5B': {'FFT': 63.3, 'LoRA_1': 66.7, 'LoRA_2': 65.3, 'LoRA_4': 65.1, 'LoRA_8': 66.5,
                     'LoRA_16': 66.3, 'LoRA_32': 66.0, 'LoRA_64': 66.4, 'LoRA_128': 65.2, 'LoRA_256': 65.1},
            '1.5B': {'FFT': 72.8, 'LoRA_1': 79.3, 'LoRA_2': 79.6, 'LoRA_4': 80.8, 'LoRA_8': 79.7,
                     'LoRA_16': 80.5, 'LoRA_32': 79.9, 'LoRA_64': 80.5, 'LoRA_128': 79.5, 'LoRA_256': 79.2},
            '3B': {'FFT': 79.2, 'LoRA_1': 81.9, 'LoRA_2': 82.3, 'LoRA_4': 81.0, 'LoRA_8': 82.3,
                   'LoRA_16': 81.9, 'LoRA_32': 82.3, 'LoRA_64': 82.2, 'LoRA_128': 81.8, 'LoRA_256': 82.3},
            '7B': {'FFT': 84.7, 'LoRA_1': 86.3, 'LoRA_2': 86.5, 'LoRA_4': 86.8, 'LoRA_8': 86.8,
                   'LoRA_16': 86.4, 'LoRA_32': 86.6, 'LoRA_64': 86.5, 'LoRA_128': 86.3, 'LoRA_256': 85.2}
        }
    }
    return data


def plot_figure1_accuracy_vs_model_size(datasets_data, dataset_name='BoolQ', 
                                        lora_ranks=[1, 2, 4, 8, 16, 32, 64, 128, 256], output_dir='figures'):
    """
    Figure 1: Accuracy vs Model Size
    
    Args:
        datasets_data: Parsed data dictionary
        dataset_name: Which dataset to plot (e.g., 'BoolQ' or 'ARC-Easy')
        lora_ranks: List of LoRA ranks to plot
        output_dir: Directory to save the figure
    """
    if dataset_name not in datasets_data:
        print(f"Dataset {dataset_name} not found. Available: {list(datasets_data.keys())}")
        return
    
    data = datasets_data[dataset_name]
    
    # Model sizes in billions (for plotting)
    model_sizes = [0.5, 1.5, 3.0, 7.0]
    
    # Model size keys in the data dictionary
    size_keys = ['0.5B', '1.5B', '3B', '7B']
    
    # Extract FFT accuracies
    fft_accuracies = [data[size_key]['FFT'] for size_key in size_keys]
    
    # Extract LoRA accuracies for each rank
    lora_data = {}
    for rank in lora_ranks:
        lora_data[rank] = [data[size_key][f'LoRA_{rank}'] for size_key in size_keys]

    # Create the plot
    plt.style.use("seaborn-v0_8-whitegrid")

    # Set serif font (Times-like) for all text and math
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Force all text to be completely black
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black'

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    # Set figure and axes frame to completely black
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    
    # Define markers for different curves (more markers for all ranks)
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', 'h', '<', '>', 'p', 'H', 'd']
    
    # Plot FFT
    ax.plot(
        model_sizes,
        fft_accuracies,
        color="black",
        marker=markers[0],
        label="Full Fine-Tuning (FFT)",
        linewidth=2.5,
        markersize=9
    )

    # Plot LoRA curves with viridis colormap (matching plot_dimension_sweep.py)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lora_ranks)))

    for i, rank in enumerate(lora_ranks):
        color = colors[i]
        marker = markers[(i + 1) % len(markers)]
        ax.plot(
            model_sizes,
            lora_data[rank],
            marker=marker,
            color=color,
            label=f"LoRA (r={rank})",
            linewidth=2,
            markersize=7
        )

    # Set log scale for x-axis
    ax.set_xscale("log")
    ax.set_xlabel("Model Size (Billion Parameters)")
    ax.set_ylabel("Accuracy (%)")
    
    # Set x-axis ticks to show model sizes
    ax.set_xticks(model_sizes)
    ax.set_xticklabels(['0.5B', '1.5B', '3B', '7B'])

    # Adjust legend for many curves - use two columns if needed
    ncol = 2 if len(lora_ranks) > 5 else 1
    ax.legend(frameon=True, labelcolor='black', ncol=ncol, loc='best')
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    
    # Save outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"figure1_accuracy_vs_model_size_{dataset_name.lower().replace('-', '_')}.png"
    pdf_path = out_dir / f"figure1_accuracy_vs_model_size_{dataset_name.lower().replace('-', '_')}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure 1 saved to {png_path} and {pdf_path}")
    return png_path, pdf_path


def plot_figure_average_accuracy_vs_model_size(datasets_data,
                                                lora_ranks=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                                                dataset_names=None,
                                                output_dir='figures'):
    """
    Figure with Average Accuracy vs Model Size

    Args:
        datasets_data: Parsed data dictionary
        lora_ranks: List of LoRA ranks to plot
        dataset_names: List of dataset names to average over. If None, averages over all datasets.
        output_dir: Directory to save the figure
    """
    model_sizes = [0.5, 1.5, 3.0, 7.0]
    size_keys = ['0.5B', '1.5B', '3B', '7B']

    # If no dataset names specified, use all datasets
    if dataset_names is None:
        dataset_names = list(datasets_data.keys())

    # Validate dataset names
    for name in dataset_names:
        if name not in datasets_data:
            print(f"Warning: Dataset {name} not found in data. Available: {list(datasets_data.keys())}")
            return

    # Calculate average accuracies across specified datasets
    average_accuracies = {size_key: {} for size_key in size_keys}

    for size_key in size_keys:
        # Get all methods for the first dataset (assuming they are all the same)
        methods = datasets_data[dataset_names[0]][size_key].keys()
        for method in methods:
            accuracies = []
            for dataset_name in dataset_names:
                accuracies.append(datasets_data[dataset_name][size_key][method])
            average_accuracies[size_key][method] = np.mean(accuracies)

    # Extract FFT accuracies
    fft_accuracies = [average_accuracies[size_key]['FFT'] for size_key in size_keys]
    
    # Extract LoRA accuracies for each rank
    lora_data = {}
    for rank in lora_ranks:
        lora_data[rank] = [average_accuracies[size_key][f'LoRA_{rank}'] for size_key in size_keys]

    # Create the plot
    plt.style.use("seaborn-v0_8-whitegrid")

    # Set serif font (Times-like) for all text and math
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Force all text to be completely black
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black'

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    # Set figure and axes frame to completely black
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    
    # Define markers for different curves (more markers for all ranks)
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', 'h', '<', '>', 'p', 'H', 'd']
    
    # Plot FFT
    ax.plot(
        model_sizes,
        fft_accuracies,
        color="black",
        marker=markers[0],
        label="Full Fine-Tuning (FFT)",
        linewidth=2.5,
        markersize=9
    )

    # Plot LoRA curves with viridis colormap (matching plot_dimension_sweep.py)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lora_ranks)))

    for i, rank in enumerate(lora_ranks):
        color = colors[i]
        marker = markers[(i + 1) % len(markers)]
        ax.plot(
            model_sizes,
            lora_data[rank],
            marker=marker,
            color=color,
            label=f"LoRA (r={rank})",
            linewidth=2,
            markersize=7
        )

    # Set log scale for x-axis
    ax.set_xscale("log")
    ax.set_xlabel("Model Size (Billion Parameters)")
    ax.set_ylabel("Average Accuracy (%)")
    
    # Set x-axis ticks to show model sizes
    ax.set_xticks(model_sizes)
    ax.set_xticklabels(['0.5B', '1.5B', '3B', '7B'])

    # Create filename suffix based on which datasets are being averaged
    if len(dataset_names) == len(datasets_data):
        filename_suffix = ""
    else:
        filename_suffix = "_" + "_".join([d.lower().replace('-', '_') for d in dataset_names])

    # Adjust legend for many curves - use two columns if needed
    ncol = 2 if len(lora_ranks) > 5 else 1
    ax.legend(frameon=True, labelcolor='black', ncol=ncol, loc='best')
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()

    # Save outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"figure_average_accuracy_vs_model_size{filename_suffix}.png"
    pdf_path = out_dir / f"figure_average_accuracy_vs_model_size{filename_suffix}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Average accuracy figure saved to {png_path} and {pdf_path}")
    return png_path, pdf_path


def plot_accuracy_vs_rank(datasets_data,
                          lora_ranks=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                          dataset_names=None,
                          model_sizes_to_average=None,
                          output_dir='figures'):
    """
    Plot Accuracy vs LoRA Rank for different model sizes

    Args:
        datasets_data: Parsed data dictionary
        lora_ranks: List of LoRA ranks to plot on x-axis
        dataset_names: List of dataset names to average over. If None, averages over all datasets.
        model_sizes_to_average: List of model sizes (in billions) to average over. If None, plots all model sizes separately.
                                Example: [0.5, 1.5] will average 0.5B and 1.5B models into a single curve.
        output_dir: Directory to save the figure
    """
    # Model sizes
    model_sizes = [0.5, 1.5, 3.0, 7.0]
    size_keys = ['0.5B', '1.5B', '3B', '7B']

    # If no dataset names specified, use all datasets
    if dataset_names is None:
        dataset_names = list(datasets_data.keys())

    # Validate dataset names
    for name in dataset_names:
        if name not in datasets_data:
            print(f"Warning: Dataset {name} not found in data. Available: {list(datasets_data.keys())}")
            return

    # Calculate average accuracies and standard deviations for each model size and rank
    # Structure: {size_key: {rank: {'mean': average_accuracy, 'std': std_dev}}}
    average_accuracies_by_size = {}

    for size_key in size_keys:
        average_accuracies_by_size[size_key] = {}
        for rank in lora_ranks:
            accuracies = []
            for dataset_name in dataset_names:
                accuracies.append(datasets_data[dataset_name][size_key][f'LoRA_{rank}'])
            average_accuracies_by_size[size_key][rank] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies)
            }

    # Calculate average FFT accuracy across specified datasets and model sizes
    # Determine which model sizes to include in FFT averaging
    if model_sizes_to_average is None:
        # Use all model sizes for FFT average
        fft_size_keys = size_keys
    else:
        # Convert specified model sizes to size_keys
        fft_size_keys = []
        for size in model_sizes_to_average:
            if size == 0.5:
                fft_size_keys.append('0.5B')
            elif size == 1.5:
                fft_size_keys.append('1.5B')
            elif size == 3 or size == 3.0:
                fft_size_keys.append('3B')
            elif size == 7 or size == 7.0:
                fft_size_keys.append('7B')

    # Compute average FFT accuracy
    fft_accuracies = []
    for dataset_name in dataset_names:
        for size_key in fft_size_keys:
            fft_accuracies.append(datasets_data[dataset_name][size_key]['FFT'])
    average_fft_accuracy = np.mean(fft_accuracies)

    # Create the plot
    plt.style.use("seaborn-v0_8-whitegrid")

    # Set serif font (Times-like) for all text and math
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Force all text to be completely black
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black'

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    # Set figure and axes frame to completely black
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    # Define markers and colors for different model sizes
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']
    colors_for_model_sizes = plt.cm.viridis(np.linspace(0.15, 0.85, 4))  # viridis colormap for 4 model sizes

    if model_sizes_to_average is None:
        # Plot all model sizes separately (current behavior)
        for i, (size_key, size_label) in enumerate(zip(size_keys, ['0.5B', '1.5B', '3B', '7B'])):
            means = [average_accuracies_by_size[size_key][rank]['mean'] for rank in lora_ranks]
            stds = [average_accuracies_by_size[size_key][rank]['std'] for rank in lora_ranks]
            ax.plot(
                lora_ranks,
                means,
                marker=markers[i],
                color=colors_for_model_sizes[i],
                label=f"Model Size: {size_label}",
                linewidth=2,
                markersize=7
            )
            # Add shaded error bars
            ax.fill_between(
                lora_ranks,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                color=colors_for_model_sizes[i],
                alpha=0.12
            )
    else:
        # Average over specified model sizes and plot single curve
        # Convert model sizes to size_keys (0.5 -> '0.5B', etc.)
        size_keys_to_average = []
        for size in model_sizes_to_average:
            if size == 0.5:
                size_keys_to_average.append('0.5B')
            elif size == 1.5:
                size_keys_to_average.append('1.5B')
            elif size == 3 or size == 3.0:
                size_keys_to_average.append('3B')
            elif size == 7 or size == 7.0:
                size_keys_to_average.append('7B')
            else:
                print(f"Warning: Model size {size} not recognized. Available: 0.5, 1.5, 3, 7")
                return

        # Validate that specified model sizes exist
        for size_key in size_keys_to_average:
            if size_key not in average_accuracies_by_size:
                print(f"Warning: Model size {size_key} not found in data.")
                return

        # Compute average accuracies and standard deviations across specified model sizes for each rank
        averaged_accuracies = []
        averaged_stds = []
        for rank in lora_ranks:
            rank_accuracies = []
            for size_key in size_keys_to_average:
                rank_accuracies.append(average_accuracies_by_size[size_key][rank]['mean'])
            averaged_accuracies.append(np.mean(rank_accuracies))
            # Standard deviation of the means across model sizes
            averaged_stds.append(np.std(rank_accuracies))

        # Use viridis color for averaged curve (matching plot_dimension_sweep.py)
        colors_lora = plt.cm.viridis(np.linspace(0.15, 0.85, len(lora_ranks)))
        curve_color = colors_lora[len(lora_ranks)//2]  # Use middle color from viridis
        ax.plot(
            lora_ranks,
            averaged_accuracies,
            marker='o',
            color=curve_color,
            label=f"LoRA",
            linewidth=2,
            markersize=7
        )
        # Add shaded error bars
        ax.fill_between(
            lora_ranks,
            np.array(averaged_accuracies) - np.array(averaged_stds),
            np.array(averaged_accuracies) + np.array(averaged_stds),
            color=curve_color,
            alpha=0.12
        )

    # Plot FFT as a horizontal constant line
    ax.axhline(
        y=average_fft_accuracy,
        color='black',
        linestyle='--',
        linewidth=2.5,
        label='Full Fine-Tuning (FFT)',
        zorder=10  # Ensure it appears on top
    )

    # Set x-axis to show all ranks
    ax.set_xlabel("LoRA Rank (r)")
    ax.set_ylabel("Accuracy (%)")

    # Set x-axis to log scale if there are many ranks
    if len(lora_ranks) > 6:
        ax.set_xscale("log")
        ax.set_xticks(lora_ranks)
        ax.set_xticklabels([str(r) for r in lora_ranks])
    else:
        ax.set_xticks(lora_ranks)

    # Create filename suffix based on which datasets are being averaged
    if len(dataset_names) == len(datasets_data):
        filename_dataset_suffix = ""
    else:
        filename_dataset_suffix = "_" + "_".join([d.lower().replace('-', '_') for d in dataset_names])

    # Add model size averaging info to filename
    if model_sizes_to_average is not None:
        size_keys_to_average = []
        for size in model_sizes_to_average:
            if size == 0.5:
                size_keys_to_average.append('0.5B')
            elif size == 1.5:
                size_keys_to_average.append('1.5B')
            elif size == 3 or size == 3.0:
                size_keys_to_average.append('3B')
            elif size == 7 or size == 7.0:
                size_keys_to_average.append('7B')

        filename_model_suffix = "_avg_" + "_".join(size_keys_to_average)
    else:
        filename_model_suffix = ""

    filename_suffix = filename_dataset_suffix + filename_model_suffix

    ax.legend(frameon=True, labelcolor='black', loc='best')
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()

    # Save outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"figure_accuracy_vs_rank{filename_suffix}.png"
    pdf_path = out_dir / f"figure_accuracy_vs_rank{filename_suffix}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Accuracy vs Rank figure saved to {png_path} and {pdf_path}")
    return png_path, pdf_path


if __name__ == "__main__":
    # Parse the table
    table_path = Path("table.tex")
    if table_path.exists():
        datasets_data = parse_table_tex_manual(table_path)
    else:
        print(f"Warning: {table_path} not found. Using hardcoded data.")
        datasets_data = parse_table_tex_manual("")
    
    # # Generate Figure 1 for BoolQ (all ranks)
    # print("Generating Figure 1 for BoolQ...")
    # plot_figure1_accuracy_vs_model_size(
    #     datasets_data,
    #     dataset_name='BoolQ',
    #     lora_ranks=[1, 4, 16, 64, 128, 256],
    #     output_dir='figures'
    # )
    
    # # Generate Figure 1 for ARC-Easy (all ranks)
    # print("Generating Figure 1 for ARC-Easy...")
    # plot_figure1_accuracy_vs_model_size(
    #     datasets_data,
    #     dataset_name='ARC-Easy',
    #     lora_ranks=[1, 4, 16, 64, 128, 256],
    #     output_dir='figures'
    # )

    # # Generate Figure 1 for ARC-Challenge (all ranks)
    # print("Generating Figure 1 for ARC-Challenge...")
    # plot_figure1_accuracy_vs_model_size(
    #     datasets_data,
    #     dataset_name='ARC-Challenge',
    #     lora_ranks=[1, 4, 16, 64, 128, 256],
    #     output_dir='figures'
    # )

    # # Generate Figure 1 for Commonsense-QA (all ranks)
    # print("Generating Figure 1 for Commonsense-QA...")
    # plot_figure1_accuracy_vs_model_size(
    #     datasets_data,
    #     dataset_name='Commonsense-QA',
    #     lora_ranks=[1, 4, 16, 64, 128, 256],
    #     output_dir='figures'
    # )
    
    # # Generate Figure for Average Accuracy across all datasets
    # print("Generating Figure for Average Accuracy (all datasets)...")
    # plot_figure_average_accuracy_vs_model_size(
    #     datasets_data,
    #     lora_ranks=[1, 4, 16, 64, 128, 256],
    #     dataset_names=None,  # None means all datasets
    #     output_dir='figures'
    # )

    # # Generate Figure for Average Accuracy vs Rank (all datasets)
    # print("Generating Figure for Accuracy vs Rank (all datasets)...")
    # plot_accuracy_vs_rank(
    #     datasets_data,
    #     lora_ranks=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    #     dataset_names=None,  # None means all datasets
    #     output_dir='figures'
    # )

    # Example: Generate Figure for specific datasets only
    print("Generating Figure for specific datasets...")
    plot_figure_average_accuracy_vs_model_size(
        datasets_data,
        lora_ranks=[1, 4, 16, 64, 128, 256],
        dataset_names=['BoolQ', 'Commonsense-QA'],
        output_dir='figures'
    )

    # Example: Generate Accuracy vs Rank for specific datasets
    # plot_accuracy_vs_rank(
    #     datasets_data,
    #     lora_ranks=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    #     dataset_names=['BoolQ', 'Commonsense-QA'],
    #     output_dir='figures'
    # )

    # Example: Generate Accuracy vs Rank with model size averaging
    print("Generating Figure for Accuracy vs Rank (averaging 0.5B and 1.5B models)...")
    plot_accuracy_vs_rank(
        datasets_data,
        lora_ranks=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        dataset_names=None,  # None means all datasets
        model_sizes_to_average=[0.5, 1.5, 3, 7],  # Average over small models
        output_dir='figures'
    )

    print("All figures generated successfully!")