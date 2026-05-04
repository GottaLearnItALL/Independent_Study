# run_all.py

from expirement_runner import run_experiment, run_transfer_experiment
from visualizations import (
    plot_individual, plot_combined, plot_cross_dataset,
    plot_transfer_comparison, print_summary_table, save_results_csv
)


# ============================================
# RUN ALL DATASETS
# ============================================

RUN_MNIST = False        # skip
RUN_CIFAR = True         # run this
RUN_EMNIST = True        # run this
RUN_TRANSFER = False      # professor's experiment

if RUN_MNIST:
    mnist_results = run_experiment('mnist')
    print_summary_table(mnist_results, 'MNIST')
    plot_combined(mnist_results, 'MNIST')
    save_results_csv(mnist_results, 'MNIST')

if RUN_CIFAR:
    cifar_results = run_experiment('cifar10')
    print_summary_table(cifar_results, 'CIFAR-10')
    plot_combined(cifar_results, 'CIFAR-10')
    save_results_csv(cifar_results, 'CIFAR-10')

if RUN_EMNIST:
    emnist_results = run_experiment('emnist')
    print_summary_table(emnist_results, 'EMNIST')
    plot_combined(emnist_results, 'EMNIST')
    save_results_csv(emnist_results, 'EMNIST')

if RUN_TRANSFER:
    mnist_pretrained_results = run_transfer_experiment()
    print_summary_table(mnist_pretrained_results, 'EMNIST (MNIST pretrained)')
    plot_transfer_comparison(emnist_results, mnist_pretrained_results, 'EMNIST')
    save_results_csv(mnist_pretrained_results, 'EMNIST_mnist_pretrained')

print("\nALL SELECTED EXPERIMENTS COMPLETE")