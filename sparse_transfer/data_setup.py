import os
from re import L
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from config import (
    IMAGENET_MEAN, IMAGENET_STD, RESNET_INPUT_SIZE, BATCH_SIZE, BATCH_SIZE_RESNET, DATASET_CONFIG
)


NUM_OF_WORKERS = os.cpu_count()


def get_transforms(dataset_name: str, for_resnet: bool =False ):
    """ 
    Build the Transforms pipeline for a given dataset.
    
    Args:
        datset_name: str
        for_resnet = bool
        for_renet = True -> resize the dataset to 244x244, convert grayscale to 3ch
        for_resnet = False -> keep the original size, just convert grayscale to 3ch
    
    Return:
        transform: torchvision.utils.transforms
    """

    config = DATASET_CONFIG[dataset_name]
    transforms_list = []

    if config['needs_grayscale_convert']:
        transforms_list.append(transforms.Grayscale(3))
    
    if for_resnet:
        transforms_list.append(transforms.Resize(RESNET_INPUT_SIZE))
    
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    return transforms.Compose(transforms_list)


def load_dataset(dataset_name, for_resnet=False):
    """ 
    Load train and test data for any supported dataset.
    Returns (train_dataset, test_dataset)
    """

    transform = get_transforms(dataset_name, for_resnet)

    if dataset_name == "mnist":
        train = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
        test = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

    elif dataset_name == "cifar10":
        train = datasets.CIFAR10(root='./data', train=True,
                               download=True, transform=transform)
        test = datasets.CIFAR10(root='./data', train=False,
                              download=True, transform=transform)
                    
    elif dataset_name == "emnist":
        train = datasets.EMNIST(root='./data', train=True,
                               download=True, transform=transform)
        test = datasets.EMNIST(root='./data', train=False,
                              download=True, transform=transform)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loaded {dataset_name} ({'resnet' if for_resnet else 'base'}) "
          f"{len(train)} train, {len(test)} test")
    return train, test


def get_sparse_subset(dataset, fraction):
    """ Randomly sample a fraction of the dataset to simulate sparse data """
    subset_size = int(len(dataset)*fraction)
    indices = np.random.choice(len(dataset), subset_size, replace=True)
    return Subset(dataset, indices)


def get_dataloader(dataset, for_resnet=False, shuffle=True):
    """ Create a Dataloader with the right batch size. """
    batch_size = BATCH_SIZE_RESNET if for_resnet else BATCH_SIZE
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


    
    

