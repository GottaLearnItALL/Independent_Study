# models.py

import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from config import DEVICE, DATASET_CONFIG


class BaseCNN(nn.Module):
    """Simple CNN with no pretrained knowledge.
    This is the control group - shows how bad things get
    without transfer learning on sparse data."""

    def __init__(self, num_classes=10, input_size=28):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        flat_size = input_size // 2 // 2
        self.flat_features = 64 * flat_size * flat_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.classifier(x)
        return x


# ============================================
# These are STANDALONE functions, NOT inside BaseCNN
# ============================================

def make_baseline(dataset_name):
    config = DATASET_CONFIG[dataset_name]
    model = BaseCNN(
        num_classes=config['num_classes'],
        input_size=config['input_size']
    )
    return model


def make_feature_extractor(dataset_name):
    config = DATASET_CONFIG[dataset_name]
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Feature Extraction — Trainable: {trainable:,} / {total:,}")

    return model


def make_full_finetune(dataset_name):
    config = DATASET_CONFIG[dataset_name]
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Full Fine-Tuning — Trainable: {trainable:,}")

    return model


def make_gradual_unfreeze(dataset_name):
    config = DATASET_CONFIG[dataset_name]
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])

    print(f"Gradual Unfreeze — Starting frozen, will unfreeze in stages")

    return model


def train_resnet_on_mnist():
    from data_setup import load_dataset, get_dataloader
    import os

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)

    train_data, test_data = load_dataset('mnist', for_resnet=True)
    train_loader = get_dataloader(train_data, for_resnet=True, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("Training ResNet18 on MNIST from scratch...")
    for epoch in tqdm(range(5)):
        model.train()
        running_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/5 — Loss: {running_loss/len(train_loader):.4f}")

    save_dir = 'results/saved_models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'resnet18_mnist.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Saved MNIST-trained weights to {save_path}")

    return model


def load_mnist_pretrained(dataset_name):
    config = DATASET_CONFIG[dataset_name]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)

    weights_path = 'results/saved_models/resnet18_mnist.pth'
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print(f"Loaded MNIST-trained weights from {weights_path}")

    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])

    return model