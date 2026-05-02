# Sparse Transfer Learning: Comparing Transfer Strategies Under Data Scarcity

**Independent Study — Rutgers University Camden**
**Advisor:** Professor Sanchirico
**Researcher:** Aryan Bhat

## Overview

This project investigates a practical question in deep learning: **when you have very limited labeled data, what is the best way to leverage a pretrained model to achieve strong classification accuracy?**

Deep learning models typically require thousands of labeled examples to perform well. In many real-world domains — medical imaging, rare species identification, manufacturing defect detection —collecting large labeled datasets is expensive, time-consuming, or impossible. Transfer learning offers a solution by reusing knowledge from models trained on large datasets (like ImageNet) and adapting them to new tasks with minimal data.

This study empirically compares three transfer learning strategies across varying levels of data scarcity, using a fixed architecture (ResNet18) and multiple benchmark datasets.

## Research Question

*"Which transfer learning strategy yields the best generalization and accuracy as labeled training data decreases?"*

## Strategies Compared

**Baseline (No Transfer Learning):** A simple CNN trained from scratch. Serves as the control group demonstrating how performance degrades without pretrained knowledge.

**Feature Extraction:** Load ResNet18 pretrained on ImageNet, freeze all layers, and train only a new final classification layer. The pretrained model acts as a fixed feature extractor. This is the safest approach with very small datasets since only ~5,000 parameters are trained.

**Full Fine-Tuning:** Load pretrained ResNet18 and allow all 11 million parameters to update during training with a reduced learning rate. This offers maximum adaptability but risks catastrophic forgetting and overfitting when data is limited.

**Gradual Unfreezing:** Start with a fully frozen pretrained ResNet18 (like feature extraction), then progressively unfreeze layers from the top down across training stages. Each stage uses a decreasing learning rate. This balances adaptability with stability.

## Datasets

**MNIST** — 60,000 handwritten digit images (28×28, grayscale, 10 classes). A baseline benchmark to validate the experimental pipeline.

**CIFAR-10** — 50,000 natural images (32×32, RGB, 10 classes). Planes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks. More complex than MNIST, making transfer learning essential.

**EMNIST Letters** — 145,000 handwritten letter images (28×28, grayscale, 27 classes). Used for cross-domain transfer experiments.

## Special Experiment: MNIST → EMNIST Transfer

Beyond using ImageNet-pretrained weights, this study also trains ResNet18 from scratch on MNIST and transfers those weights to EMNIST. This tests whether domain-similar pretraining (handwritten digits → handwritten letters) outperforms domain-diverse pretraining (ImageNet natural photos → handwritten letters).

## Project Structure

```
sparse_transfer/
├── config.py                ← All settings: device, fractions, learning rates, dataset info
├── data_utils.py            ← Dataset loading, transforms, sparse subset creation
├── models.py                ← BaseCNN, strategy builders, MNIST pretraining
├── training.py              ← Train loop, test loop, gradual unfreezing logic
├── visualization.py         ← Individual plots, combined plots, cross-dataset charts
├── experiment_runner.py     ← Master function: runs all strategies on any dataset
├── run_all.py               ← Entry point: executes all experiments
├── results/
│   ├── plots/               ← Generated comparison charts
│   ├── saved_models/        ← Saved MNIST-pretrained weights
│   ├── mnist_results.csv
│   ├── cifar10_results.csv
│   └── emnist_results.csv
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/sparse-transfer.git
cd sparse-transfer

# create virtual environment
uv venv --python 3.11
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# install dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install matplotlib numpy
```

## Usage

Run all experiments:

```bash
python run_all.py
```

This will train all 4 strategies across all sparsity levels on MNIST, CIFAR-10, and EMNIST, run the MNIST → EMNIST transfer experiment, generate comparison plots in `results/plots/`, and save numerical results to CSV files in `results/`.

## Methodology

Each experiment follows the same protocol:

1. Load the dataset and create subsets at 1%, 5%, 10%, 20%, 50%, and 100% of training data
2. For each subset size, train a fresh model using each strategy
3. Evaluate all models on the complete test set (never subsampled)
4. Record test accuracy for each strategy-fraction combination
5. Visualize results as accuracy vs. data scarcity curves

All models use Adam optimizer. Feature extraction and baseline use lr=0.001. Full fine-tuning uses lr=0.0001 to protect pretrained weights. Gradual unfreezing uses decreasing learning rates per stage (0.001 → 0.0005 → 0.0003 → 0.0001).

## Key Findings

*To be updated after experiments complete.*

Expected outcomes:

- Baseline accuracy drops sharply under data scarcity, especially on CIFAR-10
- Feature extraction provides stable performance even with very few samples
- Full fine-tuning may overfit at low data fractions but excel with more data
- Gradual unfreezing offers the best balance across all data sizes
- Domain-similar pretraining (MNIST → EMNIST) may compete with ImageNet pretraining at very low data fractions

## Technologies

- Python 3.11
- PyTorch + TorchVision
- ResNet18 (pretrained on ImageNet)
- Matplotlib
- CUDA (NVIDIA GPU acceleration)

## References

1. Pan & Yang — "A Survey on Transfer Learning" (2010)
2. Yosinski et al. — "How Transferable Are Features in Deep Neural Networks?" (2014)
3. He et al. — "Deep Residual Learning for Image Recognition" (2015)
4. Shorten & Khoshgoftaar — "A Survey on Image Data Augmentation for Deep Learning" (2019)
5. Zhuang et al. — "A Comprehensive Survey on Transfer Learning" (2020)

## License

This project is part of an academic independent study at Rutgers University Camden.