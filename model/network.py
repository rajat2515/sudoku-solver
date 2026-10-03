"""
Neural Network Architectures for Sudoku Digit Recognition in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """
    Convolutional Neural Network for recognizing handwritten and printed digits (28x28 grayscale).
    Architecture inspired by LeNet-5 with modern batch normalization and dropout.
    """

    def __init__(self, num_classes: int = 10):
        super(DigitCNN, self).__init__()
        
        # Block 1: Conv -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: Conv -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: Conv -> BN -> ReLU
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.dropout1 = nn.Dropout(0.25)
        
        # Dense Layers
        # Input size: 28x28 -> pool1 (14x14) -> pool2 (7x7) -> 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout1(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully Connected
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def build_model(num_classes: int = 10, pretrained_path: str = None, device: str = "cpu") -> DigitCNN:
    """
    Factory function to initialize and optionally load weights into DigitCNN.
    """
    model = DigitCNN(num_classes=num_classes)
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    return model
