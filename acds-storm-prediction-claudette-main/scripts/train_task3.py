# train.py
import os
import time
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from livelossplot import PlotLosses

# Custom imports
from utils.data_loader_3 import CustomImageDataset
from models.task3_model import ProbabilisticCNN_3
from utils.metrics import probabilistic_loss

def read_ids(txt_path):
    """Read storm IDs from text file"""
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]
    
def train(model, dataloader, optimizer, epoch, num_epochs, device):
    model.train()
    running_loss = 0.0

    # Progress bar for the epoch
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
        for images_batch, targets_batch in dataloader:
            images_batch = images_batch.to(device)
            targets_batch = targets_batch.to(device)

            # Forward pass
            p_zero_pred, mu_nonzero_pred, sigma_nonzero_pred = model(images_batch)

            # Compute loss
            loss = probabilistic_loss(p_zero_pred, mu_nonzero_pred, sigma_nonzero_pred, targets_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * images_batch.size(0)

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)

    # Compute average loss for the epoch
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return epoch_loss

def validate(model, dataloader, device):
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for images_batch, targets_batch in dataloader:
            images_batch = images_batch.to(device)
            targets_batch = targets_batch.to(device)

            # Forward pass
            p_zero_pred, mu_nonzero_pred, sigma_nonzero_pred = model(images_batch)

            # Compute loss
            loss = probabilistic_loss(p_zero_pred, mu_nonzero_pred, sigma_nonzero_pred, targets_batch)

            # Backward pass
            # Update running loss
            running_loss += loss.item() * images_batch.size(0)

    # Compute average loss for the epoch
    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss

def train_task3(cfg_path='configs/task3.yaml'):
    """Main training function"""
    # Load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Initialize datasets
    train_ids = read_ids(cfg['data']['train_ids_file'])
    val_ids = read_ids(cfg['data']['val_ids_file'])
    
    train_dataset = CustomImageDataset(
        csv_path=cfg['data']['csv_path'],
        h5_path=cfg['data']['h5_path'],
        storm_ids=train_ids)
    
    val_dataset = CustomImageDataset(
        csv_path=cfg['data']['csv_path'],
        h5_path=cfg['data']['h5_path'],
        storm_ids=train_ids)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True
    )

    # Initialize model
    device = cfg['training']['device']
    model = ProbabilisticCNN_3().to(device)

    # Training setup
    num_epochs = cfg['training']['epochs']
    optimizer = optim.Adam(model.parameters(), lr=float(cfg['training']['lr']))
    best_val_loss = float('inf')

    # Training loop
    liveloss = PlotLosses()
    for epoch in range(num_epochs):
        logs = {}

        # Train for one epoch
        train_loss = train(model, train_loader, optimizer, epoch, num_epochs, device)
        logs['prob loss'] = train_loss  # Training loss

        # Validate after each epoch
        val_loss = validate(model, val_loader, device)
        logs['val_prob loss'] = val_loss  # Validation loss

        # Update live loss plot
        liveloss.update(logs)
        liveloss.draw()

        # Print epoch results
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg['training']['ckpt_dir'], 'best_model.pt'))
            print("  Best model saved!")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/task3.yaml'
    train_task3(config_path)