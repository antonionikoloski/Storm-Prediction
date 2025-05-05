import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.preprocess import preprocess_channels,preprocess_vil

class Task2Dataset(Dataset):
    """
    Dataset for Task 2: VIL Prediction using pre-split train/val/test IDs.
    Loads only the 'vil' channel.
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

        self.samples = []


        self.storm_data_dict = self._preload_storms()
        self._build_subsequences()

    def _preload_storms(self):
        """
        Read each storm from HDF5 once, preprocess to shape (1, T, H, W),
        and store in a dict {storm_id: np.array(...)}.
        """
        storm_data_dict = {}
        with h5py.File(self.h5_path, 'r') as f:
            for storm_id in self.unique_storms:
                if 'vil' not in f[storm_id]:
                        print(f"[WARNING] 'vil' data not found for storm_id: {storm_id}. Skipping.")
                        continue

 
                vil_data = np.array(f[storm_id]['vil']).transpose(1, 2, 0) 
                original_min = vil_data.min()
                original_max = vil_data.max()

                vil_processed = preprocess_vil(
                    vil=vil_data,
                    resize_to=self.resize_to,
                    do_normalize=self.do_normalize
                    )
                
                vil_processed = vil_processed.transpose(0, 3, 1, 2) 
                

                storm_data_dict[storm_id] = {
                'data': vil_processed,
                'min': original_min,
                'max': original_max
                 }
        return storm_data_dict

    def _build_subsequences(self):
        """
        For each storm's pre-loaded data, slice out sub-sequences of length
        in_frames for X, and the subsequent out_frames of VIL for Y.
        Store them in self.samples as (X_tensor, Y_tensor).
        """
        for storm_id in self.unique_storms:
            full_seq_f = self.storm_data_dict.get(storm_id) 
            full_seq = full_seq_f['data'] 
            original_min = full_seq_f['min']
            original_max = full_seq_f['max']
            if full_seq is None:
                continue
            _, T, H, W = full_seq.shape
            print(f"Processing storm_id: {storm_id}, frames: {T}")
            max_start = T - (self.in_frames + self.out_frames)
            if max_start < 0:
                print(f"[WARNING] Not enough frames for storm_id: {storm_id}. Required: {self.in_frames + self.out_frames}, Available: {T}")
                continue

            start = 0
            while start <= max_start:
                in_end  = start + self.in_frames
                out_end = in_end + self.out_frames

             
                X = full_seq[:, start:in_end, :, :]  

              
                Y = full_seq[:, in_end:out_end, :, :] 
                Y = Y.squeeze(0)  


                # Convert to torch.Tensor
                X_tensor = torch.from_numpy(X).float() 
                Y_tensor = torch.from_numpy(Y).float() 

                if self.transform:
                    X_tensor = self.transform(X_tensor)

                self.samples.append((
                 X_tensor, 
                 Y_tensor, 
                 original_min, 
                 original_max  
                 ))

                start += self.stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx][0], self.samples[idx][1], self.samples[idx][2], self.samples[idx][3]
