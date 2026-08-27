"""
model_autoencoder.py -- .

Goal: Define and test the autoencoder architecture
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

data = np.load("data/dataset_normalized.npz")
train_array = data["train"]
val_array = data["val"]

class GarminDataset(torch.utils.data.Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)
    def __getitem__(self, index):
        return torch.from_numpy(self.array[index])

garmin_dataset1_train = GarminDataset(train_array)
garmin_dataset1_val = GarminDataset(val_array)

garmin_dataloader1_train = DataLoader(dataset=garmin_dataset1_train, batch_size=16, shuffle=True)
garmin_dataloader1_val = DataLoader(dataset=garmin_dataset1_val, batch_size=16) # no shuffle needed for evaluation


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
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(modele.parameters(), lr=0.001)


for i in range(50): # 10 epochs
    total_loss = 0
    for batch in garmin_dataloader1_train:
        optimizer.zero_grad()
        sortie = modele(batch)
        loss  = loss_fn(batch, sortie)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    modele.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in garmin_dataloader1_val:
            sortie = modele(batch)
            loss  = loss_fn(batch, sortie)
            val_loss += loss.item()
    print(f"eval loss -> {val_loss / len(garmin_dataloader1_val)}, and train loss -> {total_loss / len(garmin_dataloader1_train)}")
    modele.train()
