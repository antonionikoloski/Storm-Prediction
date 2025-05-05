# train.py
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from utils.data_loader_2 import StormDataset
from models.task2_model import ConvLSTMForecast
from utils.metrics import mse

def read_ids(txt_path):
    """Read storm IDs from text file"""
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def visualize_batch(batch_x, batch_y, out_dir, batch_idx=0, num_samples=4):
    """
    Visualizes and saves a batch of input and ground truth images
    """
    os.makedirs(out_dir, exist_ok=True)
    
    batch_x = batch_x.cpu().numpy()
    batch_y = batch_y.cpu().numpy()
    
    B, C, T_in, H, W = batch_x.shape
    _, T_out, _, _ = batch_y.shape
    
    num_samples = min(num_samples, B)
    
    for i in range(num_samples):
        vil_channel_idx = 3 
        if C <= vil_channel_idx:
            continue

        fig, axs = plt.subplots(2, T_in + T_out, figsize=(3*(T_in+T_out), 6))
        fig.suptitle(f"Batch {batch_idx}, Sample {i} - VIL Channel", fontsize=16)
        

        for t in range(T_in):
            axs[0, t].imshow(batch_x[i, vil_channel_idx, t], cmap='viridis', origin='upper')
            axs[0, t].set_title(f"Input T{t}")
            axs[0, t].axis("off")
        

        for t in range(T_out):
            axs[1, t].imshow(batch_y[i, t], cmap='viridis', origin='upper')
            axs[1, t].set_title(f"GT T{t}")
            axs[1, t].axis("off")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(out_dir, f"batch_{batch_idx}_sample_{i}_vil.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def train_task1b(cfg_path='configs/task1b.yaml'):
    """Main training function"""

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    train_ids = read_ids(cfg['data']['train_ids_file'])
    val_ids = read_ids(cfg['data']['val_ids_file'])
    
    train_dataset = StormDataset(
        csv_path=cfg['data']['csv_path'],
        h5_path=cfg['data']['h5_path'],
        storm_ids=train_ids,
        in_frames=cfg['data']['in_frames'],
        out_frames=cfg['data']['out_frames'],
        stride=cfg['data']['stride'],
        resize_to=tuple(cfg['data']['resize_to']),
        do_normalize=cfg['data']['do_normalize']
    )
    
    val_dataset = StormDataset(
        csv_path=cfg['data']['csv_path'],
        h5_path=cfg['data']['h5_path'],
        storm_ids=val_ids,
        in_frames=cfg['data']['in_frames'],
        out_frames=cfg['data']['out_frames'],
        stride=cfg['data']['stride'],
        resize_to=tuple(cfg['data']['resize_to']),
        do_normalize=cfg['data']['do_normalize']
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=False
    )



    model = ConvLSTMForecast(
        in_channels=cfg['model']['in_channels'],
        hidden_dim=cfg['model']['hidden_dim'],
        height=cfg['model']['height'],
        width=cfg['model']['width']
    ).to(cfg['training']['device'])

    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    best_val_loss = float('inf')

    for epoch in range(cfg['training']['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(cfg['training']['device'])
            batch_y = batch_y.to(cfg['training']['device'])
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = mse(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(cfg['training']['device'])
                batch_y = batch_y.to(cfg['training']['device'])
                pred = model(batch_x)
                val_loss += mse(pred, batch_y).item() * batch_x.size(0)
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{cfg['training']['epochs']}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(cfg['training']['ckpt_dir'], exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(cfg['training']['ckpt_dir'], 'best_model.pt')
            )
            print("--> Saved new best model")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/task2.yaml'
    train_task1b(config_path)