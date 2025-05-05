# models/task3_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

class ProbabilisticCNN_3(nn.Module):
    def __init__(self):
        super(ProbabilisticCNN_3, self).__init__()

        # Change in_channels from 3 to 4
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),  # Updated in_channels=4
            nn.BatchNorm2d(32),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(96 * 384, 128),  # Updated input size based on 384x384 pooling
            nn.ReLU(),
            #
            nn.Dropout(p=0.5),  # Add Dropout
            #
            nn.Linear(128, 3)  # Output 3 values: p_zero, mu_nonzero, sigma_nonzero
        )

    def forward(self, x):
        x = self.cnn(x)  # CNN feature extraction
        x = x.view(x.size(0), -1)  # Flatten

        # Predict probability, mean, and standard deviation
        x = self.fc(x)
        p_zero = torch.sigmoid(x[:, 0])  # Probability of zero
        mu_nonzero = x[:, 1]  # Mean for non-zero values
        sigma_nonzero = F.softplus(x[:, 2]) + 1e-6  # Ensure std dev is positive

        return p_zero, mu_nonzero, sigma_nonzero