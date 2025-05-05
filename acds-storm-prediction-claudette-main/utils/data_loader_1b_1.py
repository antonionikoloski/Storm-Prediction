import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.preprocess import preprocess_channels
class StormDatasetTask1BPreload_Production(Dataset):
    """
    Pre-load all storms from train.h5 into memory once, create sub-sequences,
    then serve them via __getitem__(). This avoids re-opening train.h5 for 
    every sub-sequence, which can be slow on large datasets.
    """
    def __init__(self,
                 csv_path,
                 h5_path,
                 storm_ids=None,
                 in_frames=12,
                 out_frames=12,
                 stride=12,
                 resize_to=(384, 384),
                 do_normalize=True,
                 transform=None):
        super().__init__()
        self.csv_df = pd.read_csv(csv_path)
        # remove 'lght' if not used
        self.csv_df = self.csv_df[self.csv_df['img_type'] != 'lght']

        if storm_ids is not None:
            self.csv_df = self.csv_df[self.csv_df['id'].isin(storm_ids)]
        self.unique_storms = self.csv_df['id'].unique()

        self.h5_path      = h5_path
        self.in_frames    = in_frames
        self.out_frames   = out_frames
        self.stride       = stride
        self.resize_to    = resize_to
        self.do_normalize = do_normalize
        self.transform    = transform

        # We will store final sub-sequences in a list
        # Each element: (X_tensor, Y_tensor)
        self.samples = []

        # 1. Pre-load all storms into memory
        self.storm_data_dict = self._preload_storms()

        # 2. For each storm, slice out sub-sequences
        self._build_subsequences()

    def _preload_storms(self):
        """
        Read each storm from HDF5 once, preprocess to shape (4, T, H, W),
        and store in a dict {storm_id: np.array(...)}.
        """
        storm_data_dict = {}
        with h5py.File(self.h5_path, 'r') as f:
            for storm_id in self.unique_storms:
                # read channels
                vis_data   = np.array(f[storm_id]['vis'])   if 'vis'   in f[storm_id] else None
                ir069_data = np.array(f[storm_id]['ir069']) if 'ir069' in f[storm_id] else None
                ir107_data = np.array(f[storm_id]['ir107']) if 'ir107' in f[storm_id] else None
                #vil_data   = np.array(f[storm_id]['vil'])   if 'vil'   in f[storm_id] else None
                # preprocess to get shape (4, 36, H, W)
                full_seq = preprocess_channels(
                    vis_data, ir069_data, ir107_data,
                    resize_to=self.resize_to,
                    do_normalize=self.do_normalize
                )
                storm_data_dict[storm_id] = full_seq
        return storm_data_dict

    def _build_subsequences(self):
        """
        For each storm's pre-loaded data, slice out sub-sequences of length
        in_frames for X, and the subsequent out_frames of VIL for Y.
        Store them in self.samples as (X_tensor, Y_tensor).
        """
        for storm_id in self.unique_storms:
            full_seq = self.storm_data_dict[storm_id]  # shape (4, T, H, W)
            T = full_seq.shape[1]  # e.g. 36 frames in time dimension
                # X: all 4 channels from start->in_end
            X = full_seq[:, 0:11, :, :]  # shape (4, in_frames, H, W)

                # Y: future vil frames from in_end->out_end
                # Assuming vil is channel index 3
            Y = full_seq[3, 0:0, :, :] # shape (out_frames, H, W)

                # Convert to torch.Tensor
            print(X.shape)    
            X_tensor = torch.from_numpy(X).float() 
            Y_tensor = torch.from_numpy(Y).float()

            if self.transform:
                X_tensor = self.transform(X_tensor)

            self.samples.append((X_tensor, Y_tensor))    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
