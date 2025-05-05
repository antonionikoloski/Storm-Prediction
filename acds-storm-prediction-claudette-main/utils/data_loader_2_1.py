import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from utils.preprocess_production import preprocess_channels
class StormDataset_Production(Dataset):
    def __init__(self, csv_path, h5_path, storm_ids, in_frames=36, out_frames=36, 
                 stride=36, resize_to=(384, 384), do_normalize=True):
        self.csv_df = pd.read_csv(csv_path)
        self.csv_df = self.csv_df[self.csv_df['id'].isin(storm_ids)]
        self.h5_path = h5_path
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.stride = stride
        self.resize_to = resize_to
        self.do_normalize = do_normalize
        self.samples = []
        
        # Preload storms
        self.storm_data = self._preload_storms()
        self._build_subsequences()

    def _preload_storms(self):
        storm_data = {}
        with h5py.File(self.h5_path, 'r') as f:
            for storm_id in self.csv_df['id'].unique():
                grp = f[str(storm_id)]
                vis = grp['vis'][()]
                ir069 = grp['ir069'][()]
                ir107 = grp['ir107'][()]
                #vil = grp['vil'][()]
                
                input_data, target_data = preprocess_channels(
                    vis, ir069, ir107,
                    self.resize_to, self.do_normalize
                )
                storm_data[storm_id] = (input_data, target_data)
        return storm_data

    def _build_subsequences(self):
        for storm_id, (input_seq, target_seq) in self.storm_data.items():
        # Input_seq shape: (3, 36, H, W) - 3 channels for all 36 frames
        # Target_seq shape: (1, 36, H, W) - VIL channel for all 36 frames
        
        # Create ONE sample per storm containing full sequence
            X = input_seq  # (3, 36, H, W) - vis/ir069/ir107 for all frames
            Y = target_seq.squeeze(0)  # (36, H, W) - VIL for all frames
        
            self.samples.append((
                torch.from_numpy(X).float(),
                torch.from_numpy(Y).float()
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]