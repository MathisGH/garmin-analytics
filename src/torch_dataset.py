"""
torch_dataset.py -- .

Goal: 
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

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

garmin_dataloader1 = DataLoader(dataset=garmin_dataset1, batch_size=16, shuffle=True)


class ModeleTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.couche = torch.nn.Linear(10, 3)

    def forward(self, x):
        return self.couche(x)


encodeur = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(1152, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 16)
)

exemple = torch.rand(16, 288, 4)
embedding = encodeur(exemple)

decodeur = torch.nn.Sequential(
    torch.nn.Linear(16, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1152)
)

reconstruction = decodeur(embedding)

print(reconstruction.shape)