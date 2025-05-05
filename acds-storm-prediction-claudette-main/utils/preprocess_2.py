import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2


def normalize_channel(channel_data, min_val=None, max_val=None):
    """Normalize channel data to [0, 1]."""
    if min_val is None:
        min_val = channel_data.min()
    if max_val is None:
        max_val = channel_data.max()
    if max_val - min_val < 1e-5:
        return np.zeros_like(channel_data)
    return (channel_data - min_val) / (max_val - min_val)

def resize_channel(channel_data, target_shape=(384, 384)):
    """Resize each frame in time dimension."""
    T = channel_data.shape[2]
    resized = []
    for t in range(T):
        frame = cv2.resize(channel_data[..., t], target_shape, interpolation=cv2.INTER_LINEAR)
        resized.append(frame[..., np.newaxis])
    return np.concatenate(resized, axis=2)

def preprocess_channels(vis, ir069, ir107, vil, resize_to, do_normalize=True):
    """Preprocess input channels (vis, ir069, ir107) and target (vil)."""
    # Resize
    vis_resized = resize_channel(vis, resize_to)
    ir069_resized = resize_channel(ir069, resize_to)
    ir107_resized = resize_channel(ir107, resize_to)
    vil_resized = resize_channel(vil, resize_to)

    # Normalize
    if do_normalize:
        # VIS: [0, 255] -> [0, 1]
        vis_norm = vis_resized / 255.0
        
        # IR channels: mean/std normalization
        ir069_norm = (ir069_resized - np.mean(ir069_resized)) / (np.std(ir069_resized) + 1e-8)
        ir107_norm = (ir107_resized - np.mean(ir107_resized)) / (np.std(ir107_resized) + 1e-8)
        
        # VIL: min-max per frame
        vil_norm = []
        for t in range(vil_resized.shape[2]):
            frame = vil_resized[..., t]
            vil_norm.append(normalize_channel(frame)[..., np.newaxis])
        vil_norm = np.concatenate(vil_norm, axis=2)
    else:
        vis_norm, ir069_norm, ir107_norm, vil_norm = vis_resized, ir069_resized, ir107_resized, vil_resized

    # Stack inputs (3, T, H, W) and target (1, T, H, W)
    input_data = np.stack([
        np.transpose(vis_norm, (2, 0, 1)),
        np.transpose(ir069_norm, (2, 0, 1)),
        np.transpose(ir107_norm, (2, 0, 1))
    ], axis=0)
    
    target_data = np.expand_dims(np.transpose(vil_norm, (2, 0, 1)), axis=0)
    
    return input_data, target_data