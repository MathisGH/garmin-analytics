"""
torch_dataset.py -- .

Goal: 
"""

import torch
import numpy as np

data = np.load("data/dataset_normalized.npz")
train_array = data["train"]

class GarminDataset(torch.utils.data.Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)
    def __getitem__(self, index):
        return torch.from_numpy(self.array[index])

garmin_dataset1 = GarminDataset(train_array)
print(len(garmin_dataset1))
print(garmin_dataset1[0].shape)