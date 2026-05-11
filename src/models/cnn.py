import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # ✅ FIX: store representation
        self._rep = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)

        self._rep = F.relu(self.fc1(x))   # store safely
        return self.fc2(self._rep)

    # ✅ FIX: required by Soteria
    def get_representation(self):
        return self._rep