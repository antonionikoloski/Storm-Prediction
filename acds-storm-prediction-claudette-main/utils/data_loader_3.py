import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from utils.preprocess_3 import stacking, multi_event_targets

class CustomImageDataset(Dataset):
    def __init__(self, csv_path, h5_path, storm_ids):
        """
        Args:
            images (torch.Tensor): Tensor of shape (num_samples, 4, 384, 384).
            targets (torch.Tensor): Tensor of shape (num_samples, 3).
        """
        self.csv_df = pd.read_csv(csv_path)
        self.csv_df = self.csv_df[self.csv_df['id'].isin(storm_ids)]
        self.h5_path = h5_path
        self.storm_data = self._preload_storms()
        self.images, self.targets = self._build_dataset()
    
    def _preload_storms(self):
        storm_data = []
        with h5py.File(self.h5_path, 'r') as f:
            for storm_id in self.csv_df['id'].unique():
                storm_id_str = str(storm_id)
                grp = f[storm_id_str]
                # event = grp['vis'][()]
                event = {img_type: grp[img_type][:] for img_type in ['vis', 'ir069', 'ir107', 'vil', 'lght']}
                storm_data.append(event)
        return storm_data
    
    def _build_dataset(self):
        scaled_image, standard_image = stacking(self.storm_data)
        targets = multi_event_targets(self.storm_data)
        return standard_image, targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        return image, target

