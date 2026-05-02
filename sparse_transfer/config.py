import torch


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


FRACTIONS  = [0.01, 0.05, 0.1, 0.2, 0.5]

BATCH_SIZE = 64 # For BaseCNN (64x64)
BATCH_SIZE_RESNET = 32 # For ResNet (244x244)

# Normalization for ImageNet
# Data Pipeline remains consistent across all expirements.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RESNET_INPUT_SIZE = (244,244)

EPOCHS = 5

#Learning Rate Per Strategy

LR_BASELINE = 0.001
LR_FEATURE_EXTRACTION = 0.01
LR_FULL_FINETUNE = 0.0001


#Gradual Unfreezing stages: (name, layers_to_unfreeze, lr, epochs)
GRADUAL_STAGES = [
    ("fc only", [], 0.001, 3),
    ("+ layer4", ["layer4"], 0.0005, 3),
    ("+ layer3", ["layer3"], 0.0003, 3),
    ("+ layer2", ["layer2"], 0.0001, 2),
]



DATASET_CONFIG = {
    'mnist': {
        'num_classes': 10,
        'input_channels': 1,
        'input_size': 28,
        'needs_grayscale_convert': True,
    },
    'cifar10': {
        'num_classes': 10,
        'input_channels': 3,
        'input_size': 32,
        'needs_grayscale_convert': False,
    },
    'emnist': {
        'num_classes': 27,
        'input_channels': 1,
        'input_size': 28,
        'needs_grayscale_convert': True,
    },
}

# OUTPUT PATHS
RESULTS_DIR = 'results'
PLOTS_DIR = 'results/plots'
MODEL_DIR = 'results/saved_models'