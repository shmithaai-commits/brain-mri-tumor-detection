import torch
import torch.nn as nn
import torchvision.models as models

class TumorClassifier(nn.Module):
    """
    ResNet18-based classifier.
    Predicts: 0 = No Tumor, 1 = Tumor Present
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.base.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.base(x)
