
import torch

from model import get_number_net
from datasets import get_dataset
import torch.optim as optim
import torch.nn as nn
train_dataset, test_dataset = get_dataset()

for idx, data in enumerate(test_dataset):
    inputs, labels = data
    print(inputs, labels)
    break