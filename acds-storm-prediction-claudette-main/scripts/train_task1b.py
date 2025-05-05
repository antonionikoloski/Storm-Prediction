import os
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from utils.data_loader import StormDatasetTask1BPreload 
from models.task1b_model import Task1bConvLSTM 
from models.task1b_model_attention import Task1bEnhancedConvLSTM 
from utils.metrics import mse  
import pandas as pd
import matplotlib.pyplot as plt

def read_ids(txt_path):
    with open(txt_path, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    return ids

def visualize_batch(batch_x, batch_y, out_dir, batch_idx=0, num_samples=4):
    """
    Visualizes and saves a batch of input and ground truth images.

    Parameters:
    - batch_x (torch.Tensor): Input tensor of shape (B, C, T_in, H, W)
    - batch_y (torch.Tensor): Ground truth tensor of shape (B, T_out, H, W)
    - out_dir (str): Directory to save the visualization
    - batch_idx (int): Index of the batch (used in the filename)
    - num_samples (int): Number of samples from the batch to visualize
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
            print(f"[ERROR] Expected at least {vil_channel_idx + 1} channels, but got {C}.")
            continue
        
 
        num_cols = T_in + T_out
        fig, axs = plt.subplots(2, num_cols, figsize=(3*num_cols, 6))
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
        print(f"[INFO] Saved VIL visualization at: {plot_path}")


def train_task1b(cfg_path='configs/task1b.yaml'):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    
    train_ids = read_ids(cfg['data']['train_ids_file'])
    val_ids   = read_ids(cfg['data']['val_ids_file'])    
    print(f"Training on {len(train_ids)} storms, validating on {len(val_ids)} storms.")
    train_dataset = StormDatasetTask1BPreload(
        csv_path=cfg['data']['csv_path'],
        h5_path=cfg['data']['h5_path'],
        storm_ids=train_ids,
        in_frames=cfg['data']['in_frames'],
        out_frames=cfg['data']['out_frames'],
        stride=cfg['data']['stride'],
        resize_to=tuple(cfg['data']['resize_to']),
        do_normalize=cfg['data']['do_normalize']
    )
    val_dataset = StormDatasetTask1BPreload(
        csv_path=cfg['data']['csv_path'],
        h5_path=cfg['data']['h5_path'],
        storm_ids=val_ids,
        in_frames=cfg['data']['in_frames'],
        out_frames=cfg['data']['out_frames'],
        stride=cfg['data']['stride'],
        resize_to=tuple(cfg['data']['resize_to']),
        do_normalize=cfg['data']['do_normalize']
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg['training']['batch_size'], shuffle=False)
    visualization_dir = 'visualizations/train'
    try:
        # Get the first batch from the train_loader
        batch_x, batch_y = next(iter(train_loader))
        visualize_batch(batch_x, batch_y, out_dir=visualization_dir, batch_idx=0, num_samples=4)
    except Exception as e:
        print(f"[ERROR] Failed to visualize training batch: {e}")


    model = Task1bConvLSTM(
        in_channels=cfg['model']['in_channels'],  
        in_time=cfg['data']['in_frames'],         
        out_frames=cfg['data']['out_frames']       
    )
    model = model.to(cfg['training']['device'])


    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    criterion = mse 


    best_val_loss = float('inf')
    for epoch in range(cfg['training']['epochs']):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(cfg['training']['device'])
            batch_y = batch_y.to(cfg['training']['device'])

            optimizer.zero_grad()
            pred = model(batch_x) 

            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(cfg['training']['device'])
                batch_y = batch_y.to(cfg['training']['device'])

                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{cfg['training']['epochs']}], "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(cfg['training']['ckpt_dir'], exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(cfg['training']['ckpt_dir'], 'best_model.pt'))
            print("  --> Saved best model. on path" + os.path.join(cfg['training']['ckpt_dir'], 'best_model.pt'))

if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/task1b.yaml'
    train_task1b(cfg_path)
