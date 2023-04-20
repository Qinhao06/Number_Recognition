import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, ], [0.5, ])])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return train_dataset, test_dataset
