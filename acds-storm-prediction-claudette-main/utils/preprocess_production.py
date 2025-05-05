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

def preprocess_channels(vis, ir069, ir107, resize_to, do_normalize=True):
    """Preprocess input channels (vis, ir069, ir107) and target (vil)."""
    # Resize
    vis_resized = resize_channel(vis, resize_to)
    ir069_resized = resize_channel(ir069, resize_to)
    ir107_resized = resize_channel(ir107, resize_to)

    # Normalize
 

    # Stack inputs (3, T, H, W) and target (1, T, H, W)
    input_data = np.stack([
        np.transpose(vis_resized, (2, 0, 1)),
        np.transpose(ir069_resized, (2, 0, 1)),
        np.transpose(ir107_resized, (2, 0, 1))
    ], axis=0)
    
    target_data = np.zeros((1, vis_resized.shape[2], vis_resized.shape[0], vis_resized.shape[1]))
    
    return input_data, target_data