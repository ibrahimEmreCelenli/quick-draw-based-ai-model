import torch.nn as nn
from torchvision import models

class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes=345):
        super(QuickDrawCNN, self).__init__()
        
        # Load a standard ResNet18
        # We start from scratch (weights=None) because:
        # 1. We have a lot of data (1.7M images)
        # 2. Our images are 28x28 grayscale (ImageNet is 224x224 RGB)
        self.model = models.resnet18(weights=None)
        
        # Modify the first Convolutional Layer to accept 1 channel (Grayscale) instead of 3 (RGB)
        # Original ResNet input: (3, 64, kernel=7, stride=2, padding=3)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final Fully Connected layer to output our number of classes (345)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)
