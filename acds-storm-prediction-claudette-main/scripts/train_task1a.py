import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.data_loader_1a import Task2Dataset
from models.task1a_model import EnhancedVILPredictor
from utils.metrics import mse  #
from utils.preprocess import denormalize

def read_ids(txt_path):
    """
    Reads a list of IDs from a text file.

    Args:
        txt_path (str): Path to the text file containing IDs.

    Returns:
        list: List of IDs as strings.
    """
    with open(txt_path, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    return ids

def visualize_batch(batch_x, batch_y, out_dir, batch_idx=0, num_samples=4):
    """
    Visualizes and saves a batch of input and ground truth images.

    Parameters:
    - batch_x (torch.Tensor): Input tensor of shape (B, 1, T_in, H, W)
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
        num_cols = T_in + T_out
        fig, axs = plt.subplots(2, num_cols, figsize=(3*num_cols, 6))
        fig.suptitle(f"Batch {batch_idx}, Sample {i} - VIL Channel", fontsize=16)
        
       
        for t in range(T_in):
            axs[0, t].imshow(batch_x[i, 0, t], cmap='viridis', origin='upper') 
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

def train_task2(cfg_path='configs/task1a.yaml'):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    train_ids = read_ids(cfg['data']['train_ids_file'])
    val_ids = read_ids(cfg['data']['val_ids_file'])
    print(f"Training on {len(train_ids)} storms, validating on {len(val_ids)} storms.")

    train_dataset = Task2Dataset(
        csv_path=cfg['data']['csv_path'],
        h5_path=cfg['data']['h5_path'],
        storm_ids=train_ids,
        in_frames=cfg['data']['in_frames'],
        out_frames=cfg['data']['out_frames'],
        stride=cfg['data']['stride'],
        resize_to=tuple(cfg['data']['resize_to']),
        do_normalize=cfg['data']['do_normalize']
    )
    val_dataset = Task2Dataset(
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

    model = EnhancedVILPredictor(
        input_channels=cfg['model']['in_channels'],
        hidden_dim=cfg['model']['hidden_dim'],
        out_frames=cfg['data']['out_frames']
    )
    model = model.to(cfg['training']['device'])

    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    criterion = nn.L1Loss()  


    best_val_loss = float('inf')
    os.makedirs(cfg['training']['ckpt_dir'], exist_ok=True)

    for epoch in range(cfg['training']['epochs']):
        model.train()
        train_loss = 0.0
        train_loss_denorm = 0.0
        for batch_idx, (inputs, targets, mins, maxs) in enumerate(train_loader):
            inputs = inputs.to(cfg['training']['device'])  
            targets = targets.to(cfg['training']['device']) 

            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs.squeeze(2), targets[:, :cfg['data']['out_frames']])  
            outputs = outputs.squeeze(2)  
            mins = mins.to(cfg['training']['device']) 
            maxs = maxs.to(cfg['training']['device'])

            outputs_denorm = outputs * (maxs - mins).view(-1, 1, 1, 1) + mins.view(-1, 1, 1, 1)
            targets_denorm = targets * (maxs - mins).view(-1, 1, 1, 1) + mins.view(-1, 1, 1, 1)
        
            loss_denorm = criterion(outputs_denorm, targets_denorm)
            train_loss_denorm += loss_denorm.item() * inputs.size(0)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_loss_denorm /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        val_loss_denorm = 0.0
        with torch.no_grad():
            for inputs, targets, mins, maxs in val_loader:
                inputs = inputs.to(cfg['training']['device'])
                targets = targets.to(cfg['training']['device'])
                mins = mins.to(cfg['training']['device']) 
                maxs = maxs.to(cfg['training']['device'])
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(2), targets[:, :cfg['data']['out_frames']])  
               
                val_loss += loss.item() * inputs.size(0)
                outputs = outputs.squeeze(2)  # Shape: (B=2, T_out=12, H=384, W=384)
                outputs_denorm = outputs * (maxs - mins).view(-1, 1, 1, 1) + mins.view(-1, 1, 1, 1)
                targets_denorm = targets * (maxs - mins).view(-1, 1, 1, 1) + mins.view(-1, 1, 1, 1)
        
                loss_denorm = criterion(outputs_denorm, targets_denorm)
                val_loss_denorm += loss_denorm.item() * inputs.size(0)
                val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{cfg['training']['epochs']}], "
      f"Train Loss: {train_loss:.4f}, Train Loss: {train_loss_denorm:.4f},  Val Loss (Norm): {val_loss:.4f}, Val Loss (Original): {val_loss_denorm:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(cfg['training']['ckpt_dir'], 'best_model.pt'))
            print("  --> Saved best model.")

if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/task1a.yaml'
    train_task2(cfg_path)
