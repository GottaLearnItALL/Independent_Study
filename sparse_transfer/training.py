 # training.py

import torch
import torch.nn as nn
from config import DEVICE, EPOCHS, GRADUAL_STAGES
from tqdm import tqdm


def train_model(model, train_loader, epochs=EPOCHS, lr=0.001):
    """
    Standard training loop. Works for baseline, feature extraction,
    and full fine-tuning. The model determines what gets trained
    based on which parameters have requires_grad=True.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (output.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total * 100
        avg_loss = running_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%")

    return model


def test_model(model, test_loader):
    """
    Evaluate model on the full test set.
    Returns accuracy as a percentage.
    """
    model = model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = model(X)
            correct += (output.argmax(1) == y).sum().item()
            total += y.size(0)

    accuracy = correct / total * 100
    return accuracy


def train_gradual_unfreeze(model, train_loader, test_loader):
    """
    Strategy 3: Gradual Unfreezing

    The model starts fully frozen (like feature extraction), then we progressively unfreeze layers
    from top to bottom during training.

    Stage 1: train only fc           (same as feature extraction)
    Stage 2: unfreeze layer4, train  (adapting high-level features)
    Stage 3: unfreeze layer3, train  (adapting mid-level features)
    Stage 4: unfreeze layer2, train  (adapting low-level features)

    Learning rate decreases each stage — deeper layers need
    gentler updates because their features are more general
    and more valuable to preserve.
    """
    model = model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    for stage_name, layer_names, lr, epochs in GRADUAL_STAGES:

        # unfreeze the specified layers
        for name in layer_names:
            layer = getattr(model, name)  # converts string "layer4" to model.layer4
            for param in layer.parameters():
                param.requires_grad = True

        # new optimizer each stage because trainable params changed
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

        # count trainable params at this stage
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Stage: {stage_name} — LR: {lr}, Epochs: {epochs}, "
              f"Trainable: {trainable:,}")

        for epoch in tqdm(range(epochs), desc=f"Stage: {stage_name}"):
            model.train()
            running_loss = 0
            correct = 0
            total = 0

            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                output = model(X)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct += (output.argmax(1) == y).sum().item()
                total += y.size(0)

            train_acc = correct / total * 100
            avg_loss = running_loss / len(train_loader)
            print(f"    Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%")

    # test after all stages complete
    accuracy = test_model(model, test_loader)
    return accuracy