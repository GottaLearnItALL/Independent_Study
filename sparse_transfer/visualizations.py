# visualization.py

import matplotlib.pyplot as plt
import os
from config import FRACTIONS, PLOTS_DIR, RESULTS_DIR


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


COLORS = {
    'Baseline': 'red',
    'Feature Extraction': 'blue',
    'Full Fine-Tuning': 'green',
    'Gradual Unfreezing': 'purple'
}

MARKERS = {
    'Baseline': 'o',
    'Feature Extraction': 's',
    'Full Fine-Tuning': '^',
    'Gradual Unfreezing': 'D'
}


def plot_individual(results, strategy_name, dataset_name):
    """
    Plot accuracy vs data scarcity for a single strategy.
    One line, one chart.
    """
    ensure_dirs()
    percentages = [f * 100 for f in FRACTIONS]
    accuracies = [results[f] for f in FRACTIONS]

    plt.figure(figsize=(8, 5))
    plt.plot(percentages, accuracies,
             f'{MARKERS[strategy_name]}-',
             linewidth=2,
             color=COLORS[strategy_name])
    plt.xlabel('Training Data Used (%)')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'{strategy_name} — {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{strategy_name.lower().replace(' ', '_')}_{dataset_name.lower()}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.show()


def plot_combined(all_results, dataset_name):
    """
    Plot all strategies on one chart for a single dataset.
    This is your money shot — the main comparison chart.
    """
    ensure_dirs()
    percentages = [f * 100 for f in FRACTIONS]

    plt.figure(figsize=(10, 6))
    for strategy_name, results in all_results.items():
        accuracies = [results[f] for f in FRACTIONS]
        plt.plot(percentages, accuracies,
                 f'{MARKERS[strategy_name]}-',
                 linewidth=2,
                 color=COLORS[strategy_name],
                 label=strategy_name)

    plt.xlabel('Training Data Used (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'All Strategies Compared — {dataset_name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"all_strategies_{dataset_name.lower()}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.show()


def plot_cross_dataset(all_dataset_results, strategy_name):
    """
    Compare one strategy across multiple datasets.
    Shows how the same strategy performs on easy vs hard tasks.
    """
    ensure_dirs()
    percentages = [f * 100 for f in FRACTIONS]

    dataset_colors = {
        'MNIST': 'blue',
        'CIFAR-10': 'red',
        'EMNIST': 'green'
    }

    plt.figure(figsize=(10, 6))
    for dataset_name, results in all_dataset_results.items():
        accuracies = [results[f] for f in FRACTIONS]
        plt.plot(percentages, accuracies, 'o-',
                 linewidth=2,
                 color=dataset_colors.get(dataset_name, 'black'),
                 label=dataset_name)

    plt.xlabel('Training Data Used (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'{strategy_name} — Cross Dataset Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"cross_dataset_{strategy_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.show()


def plot_transfer_comparison(imagenet_results, mnist_results, dataset_name):
    """
    Compare ImageNet-pretrained vs MNIST-pretrained transfer to EMNIST.
    This is the professor's special experiment.
    """
    ensure_dirs()
    percentages = [f * 100 for f in FRACTIONS]

    plt.figure(figsize=(10, 6))
    for strategy_name in imagenet_results:
        accuracies = [imagenet_results[strategy_name][f] for f in FRACTIONS]
        plt.plot(percentages, accuracies,
                 f'{MARKERS[strategy_name]}-',
                 linewidth=2,
                 color=COLORS[strategy_name],
                 label=f'{strategy_name} (ImageNet)')

    for strategy_name in mnist_results:
        accuracies = [mnist_results[strategy_name][f] for f in FRACTIONS]
        plt.plot(percentages, accuracies,
                 f'{MARKERS[strategy_name]}--',  # dashed for MNIST pretrained
                 linewidth=2,
                 color=COLORS[strategy_name],
                 alpha=0.6,
                 label=f'{strategy_name} (MNIST pretrained)')

    plt.xlabel('Training Data Used (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'ImageNet vs MNIST Pretrained — {dataset_name}', fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"transfer_comparison_{dataset_name.lower()}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.show()


def print_summary_table(all_results, dataset_name):
    """
    Print a clean table of all results.
    Good for your writeup.
    """
    strategies = list(all_results.keys())

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY — {dataset_name}")
    print(f"{'='*70}")

    # header
    header = f"{'Data %':<10}"
    for s in strategies:
        header += f"{s:<20}"
    print(header)
    print("-" * (10 + 20 * len(strategies)))

    # rows
    for frac in FRACTIONS:
        row = f"{frac*100:<10.0f}"
        for s in strategies:
            row += f"{all_results[s][frac]:<20.2f}"
        print(row)


def save_results_csv(all_results, dataset_name):
    """
    Save results to CSV for later analysis.
    """
    ensure_dirs()
    filepath = os.path.join(RESULTS_DIR, f"{dataset_name.lower()}_results.csv")

    strategies = list(all_results.keys())

    with open(filepath, 'w') as f:
        # header
        f.write("fraction," + ",".join(strategies) + "\n")
        # rows
        for frac in FRACTIONS:
            values = [str(all_results[s][frac]) for s in strategies]
            f.write(f"{frac}," + ",".join(values) + "\n")

    print(f"Saved results to {filepath}")