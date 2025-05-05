# models/task3_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt
import os


import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 
            4 * hidden_dim, 
            kernel_size, 
            padding=kernel_size//2
        )
    
    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, gates.shape[1]//4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
class ConvLSTMForecast(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32, height=384, width=384, kernel_size=3):
        super().__init__()
        # Encoder ConvLSTM
        self.hidden_dim = hidden_dim
        


        self.encoder_lstm = ConvLSTMCell(in_channels, hidden_dim)
        
        # Output layer (predict VIL for each timestep)
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)  # 1 channel for VIL

    def forward(self, x):
        # x shape: (B, C=3, T=36, H, W)
        B, C, T, H, W = x.shape
        device = x.device
        
        # Initialize hidden/cell states
        h = torch.zeros(B, self.hidden_dim, H, W, device=device)
        c = torch.zeros_like(h)
        
        outputs = []
        for t in range(T):
            # Process each timestep
            x_t = x[:, :, t]  # (B, 3, H, W)
            h, c = self.encoder_lstm(x_t, h, c)
            
            # Predict VIL for current timestep
            vil_pred = self.output_conv(h)  # (B, 1, H, W)
            outputs.append(vil_pred.squeeze(1))  # (B, H, W)
        
        return torch.stack(outputs, dim=1)  # (B, T=36, H, W)