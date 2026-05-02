from config import (
    DEVICE, FRACTIONS, EPOCHS,
    LR_BASELINE, LR_FEATURE_EXTRACTION, LR_FULL_FINETUNE,
    DATASET_CONFIG
)
from data_setup import load_dataset, get_sparse_subset, get_dataloader
from models import (
    make_baseline, make_feature_extractor, make_full_finetune,
    make_gradual_unfreeze, train_resnet_on_mnist, load_mnist_pretrained
)
from training import train_model, test_model, train_gradual_unfreeze

def run_experiment(dataset_name):
    """
    Run all 4 strategies across all sparsity levels for one dataset.
    Returns a dictionary of results.

    This is the master function — one call does everything:
    loads data, trains models, tests them, returns results.
    """
    config = DATASET_CONFIG[dataset_name]
    num_classes = config['num_classes']
    print("Independent Study- Aryan Bhat\n\n")
    print(f"EXPERIMENT: {dataset_name.upper()}")
    print(f"Classes: {num_classes}")
    print(f"Fractions: {FRACTIONS}")
    print(f"Device: {DEVICE}")
    print("\n")


    # load data — base size for BaseCNN, resnet size for transfer strategies
    train_base, test_base = load_dataset(dataset_name, for_resnet=False)
    train_resnet, test_resnet = load_dataset(dataset_name, for_resnet=True)

    test_loader_base = get_dataloader(test_base, for_resnet=False, shuffle=False)
    test_loader_resnet = get_dataloader(test_resnet, for_resnet=True, shuffle=False)

    all_results = {
        'Baseline': {},
        'Feature Extraction': {},
        'Full Fine-Tuning': {},
        'Gradual Unfreezing': {}
    }

    for frac in FRACTIONS:
        samples = int(len(train_base) * frac)
        print(f"\n{'-'*30}")
        print(f"  {frac*100}% DATA — {samples} samples")
        print(f"{'-'*30}")

        # ----- Baseline -----
        print("\n Baseline (No Transfer Learning \n) ")
        model = make_baseline(dataset_name)
        subset = get_sparse_subset(train_base, frac)
        loader = get_dataloader(subset, for_resnet=False, shuffle=True)
        train_model(model, loader, epochs=EPOCHS, lr=LR_BASELINE)
        acc = test_model(model, test_loader_base)
        all_results['Baseline'][frac] = acc
        print(f"\n>>> TEST ACCURACY: {acc:.2f}%")

        # ----- Feature Extraction -----
        print("\nFeature Extraction (Frozen ResNet18) \n")
        model = make_feature_extractor(dataset_name)
        subset = get_sparse_subset(train_resnet, frac)
        loader = get_dataloader(subset, for_resnet=True, shuffle=True)
        train_model(model, loader, epochs=EPOCHS, lr=LR_FEATURE_EXTRACTION)
        acc = test_model(model, test_loader_resnet)
        all_results['Feature Extraction'][frac] = acc
        print(f"\n >>> TEST ACCURACY: {acc:.2f}%")

        # ----- Full Fine-Tuning -----
        print("\nFull Fine-Tuning (Unfrozen ResNet18) \n")
        model = make_full_finetune(dataset_name)
        subset = get_sparse_subset(train_resnet, frac)
        loader = get_dataloader(subset, for_resnet=True, shuffle=True)
        train_model(model, loader, epochs=EPOCHS, lr=LR_FULL_FINETUNE)
        acc = test_model(model, test_loader_resnet)
        all_results['Full Fine-Tuning'][frac] = acc
        print(f"\n>>> TEST ACCURACY: {acc:.2f}%")

        # ----- Gradual Unfreezing -----
        print("\nGradual Unfreezing (Progressive ResNet18) \n")
        model = make_gradual_unfreeze(dataset_name)
        subset = get_sparse_subset(train_resnet, frac)
        loader = get_dataloader(subset, for_resnet=True, shuffle=True)
        acc = train_gradual_unfreeze(model, loader, test_loader_resnet)
        all_results['Gradual Unfreezing'][frac] = acc
        print(f"\n>>> TEST ACCURACY: {acc:.2f}%")

    return all_results


def run_transfer_experiment():
    """
    Train ResNet18 on MNIST, then transfer to EMNIST.
    Compare against ImageNet-pretrained transfer to EMNIST.
    """
    print(f"\n{'-'*60}")
    print(f"  SPECIAL EXPERIMENT: MNIST → EMNIST TRANSFER")
    print(f"{'-'*60}")

    # Step 1: Train ResNet on MNIST and save weights
    print("\n Step 1: Training ResNet18 on MNIST \n")
    train_resnet_on_mnist()

    # Step 2: Load EMNIST data
    train_resnet, test_resnet = load_dataset('emnist', for_resnet=True)
    test_loader = get_dataloader(test_resnet, for_resnet=True, shuffle=False)

    mnist_pretrained_results = {
        'Feature Extraction': {},
        'Full Fine-Tuning': {},
        'Gradual Unfreezing': {}
    }

    for frac in FRACTIONS:
        samples = int(len(train_resnet) * frac)
        print(f"\n{'-'*30}")
        print(f"  {frac*100}% DATA — {samples} samples")
        print(f"{'-'*30}")

        # Feature Extraction with MNIST weights
        print("\n Feature Extraction (MNIST pretrained) \n")
        model = load_mnist_pretrained('emnist')
        for param in model.parameters():
            param.requires_grad = False
        model.fc.requires_grad_(True)
        subset = get_sparse_subset(train_resnet, frac)
        loader = get_dataloader(subset, for_resnet=True, shuffle=True)
        train_model(model, loader, epochs=EPOCHS, lr=LR_FEATURE_EXTRACTION)
        acc = test_model(model, test_loader)
        mnist_pretrained_results['Feature Extraction'][frac] = acc
        print(f">>> TEST ACCURACY: {acc:.2f}%")

        # Full Fine-Tuning with MNIST weights
        print("\n--- Full Fine-Tuning (MNIST pretrained) ---")
        model = load_mnist_pretrained('emnist')
        subset = get_sparse_subset(train_resnet, frac)
        loader = get_dataloader(subset, for_resnet=True, shuffle=True)
        train_model(model, loader, epochs=EPOCHS, lr=LR_FULL_FINETUNE)
        acc = test_model(model, test_loader)
        mnist_pretrained_results['Full Fine-Tuning'][frac] = acc
        print(f">>> TEST ACCURACY: {acc:.2f}%")

        # Gradual Unfreezing with MNIST weights
        print("\n Gradual Unfreezing (MNIST pretrained) \n")
        model = load_mnist_pretrained('emnist')
        for param in model.parameters():
            param.requires_grad = False
        model.fc.requires_grad_(True)
        subset = get_sparse_subset(train_resnet, frac)
        loader = get_dataloader(subset, for_resnet=True, shuffle=True)
        acc = train_gradual_unfreeze(model, loader, test_loader)
        mnist_pretrained_results['Gradual Unfreezing'][frac] = acc
        print(f">>> TEST ACCURACY: {acc:.2f}%")

    return mnist_pretrained_results