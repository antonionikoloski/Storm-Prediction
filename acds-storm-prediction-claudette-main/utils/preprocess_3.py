import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2

# --------------------------- Data Preprocessing ----------------------------------

def enlarging_ir(event_ir):
  # Assuming event['ir069'] is of shape (192, 192, 36)
  array = torch.tensor(event_ir, dtype=torch.float32)  # Convert to PyTorch tensor

  # Permute to (Samples, 1, Height, Width) to make it compatible with F.interpolate
  array = array.permute(2, 0, 1).unsqueeze(1)  # Shape: (36, 1, 192, 192)

  # Resize using bilinear interpolation
  resized_ir = F.interpolate(array, size=(384, 384), mode='bilinear', align_corners=False)

  # Remove channel dimension if necessary
  resized_ir = resized_ir.squeeze(1)  # Shape: (36, 384, 384)
  return resized_ir

def stacking(event_list):
  scales = []
  standardized = []

  for event in event_list:
    array1 = torch.tensor(event['vis'].transpose(2, 0, 1))
    array2 = enlarging_ir(event['ir069'])
    array3 = enlarging_ir(event['ir107'])
    array4 = torch.tensor(event['vil'].transpose(2,0,1))

    # Stack along the channel dimension (dim=1)
    stacked_array = torch.stack([array1, array2, array3, array4], dim=1)  # Shape: (36, 4, 384, 384)

    # Compute mean and std for each channel per sample
    mean = stacked_array.mean(dim=(2, 3), keepdim=True)  # Mean along spatial dimensions (384, 384)
    std = stacked_array.std(dim=(2, 3), keepdim=True)    # Std along spatial dimensions (384, 384)

    # Compute min and max for each channel per sample
    min_vals, _ = stacked_array.min(dim=2, keepdim=True)  # Min along height (384)
    min_vals, _ = min_vals.min(dim=3, keepdim=True)       # Min along width (384)

    max_vals, _ = stacked_array.max(dim=2, keepdim=True)  # Max along height (384)
    max_vals, _ = max_vals.max(dim=3, keepdim=True)       # Max along width (384)

    # Standardize the tensor
    standardized_array = (stacked_array - mean) / (std + 1e-6)  # Add epsilon to avoid division by zero

    # Min-Max scaling
    scaled_array = (stacked_array - min_vals) / (max_vals - min_vals + 1e-6)  # Add epsilon to avoid division by zero

    # Multiply by 255 to bring values to 0-255 range
    scaled_array *= 255

    scales.append(scaled_array)
    standardized.append(standardized_array)

  # return scales, standardized
  return torch.cat(scales, dim=0), torch.cat(standardized, dim=0)

# scales_train, standard_train = stacking(event_train)
# scales_val, standard_val = stacking(event_val)

# --------------------------- Target Preprocessing --------------------------------
def get_distribs(event):
  distribs = []

  for ti in range(36):
    t = event["lght"][:,0]
    f = (t >= ti*5*60 - 2.5*60) & (t < ti*5*60 + 2.5*60)
    first = np.zeros(int(ti*5*60 + 2.5*60 - (ti*5*60 - 2.5*60)))
    for i in range(int(ti*5*60 + 2.5*60 - (max(ti*5*60 - 2.5*60,0)))):
      first[i] = np.sum(event["lght"][f,:][:, 0] == i + max(ti*5*60 - 2.5*60,0))
    distribs.append(first)
  return distribs

def get_targets(distribs):

  targets = np.zeros((36,3))

  for i,dist in enumerate(distribs):
    dist_array=np.array(dist)
    targets[i][0] = (np.sum(dist_array == 0) / len(dist_array))
    if targets[i][0] == 1:
      targets[i][1] = 0
      targets[i][2] = 0
    else:
      targets[i][1] = np.mean(dist_array[dist_array != 0])
      targets[i][2] = np.std(dist_array[dist_array != 0])

  return torch.tensor(targets)

def multi_event_targets(event_list):
  targets = []
  for event in event_list:
    distribs = get_distribs(event)
    targets.append(get_targets(distribs))
  return torch.cat(targets, dim=0)

# targets_train = multi_event_targets(event_train)
# targets_val = multi_event_targets(event_val)