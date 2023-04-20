import torch.nn as nn


class numberNet(nn.Module):
    def __init__(self, image_size=28 * 28):
        super(numberNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, (2, 2))
        self.conv2 = nn.Conv2d(3, 1, (3, 3))
        self.fc1 = nn.Linear(25 * 25, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        # print(x.size())
        x = x.view(1, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


def get_number_net():
    return numberNet()
