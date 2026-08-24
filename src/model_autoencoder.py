"""
model_autoencoder.py -- .

Goal: Define and test the autoencoder architecture
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



class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encodeur = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(1152, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 16)
                )
        self.decodeur = torch.nn.Sequential(
                    torch.nn.Linear(16, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 1152)
                )
    def forward(self, x):
        embedding = self.encodeur(x)
        decoded = self.decodeur(embedding)
        result = decoded.view(-1, 288, 4)

        return result

modele = Autoencoder()
for batch in garmin_dataloader1:
    sortie = modele(batch)
    print(batch.shape, sortie.shape)
    break

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(modele.parameters(), lr=0.001)