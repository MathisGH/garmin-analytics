"""
model_autoencoder.py -- .

Goal: Define and test the autoencoder architecture
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch


class GarminDataset(torch.utils.data.Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)
    def __getitem__(self, index):
        return torch.from_numpy(self.array[index])


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

if __name__ == "__main__":
    data = np.load("data/dataset_normalized.npz")
    train_array = data["train"]
    eval_array = data["val"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db") # mlflow ui --backend-store-uri sqlite:///mlflow.db
    mlflow.set_experiment("autoencoder")

    garmin_dataset1_train = GarminDataset(train_array)
    garmin_dataset1_eval = GarminDataset(eval_array)

    garmin_dataloader1_train = DataLoader(dataset=garmin_dataset1_train, batch_size=16, shuffle=True)
    garmin_dataloader1_eval = DataLoader(dataset=garmin_dataset1_eval, batch_size=16) # no shuffle needed for evaluation


    model = Autoencoder()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": 0.001,
            "batch_size": 16,
            "n_epochs": 40,
        })
        # Model training
        for epoch in range(40): # 10 epochs
            total_loss = 0
            for batch in garmin_dataloader1_train:
                optimizer.zero_grad()
                sortie = model(batch)
                loss  = loss_fn(batch, sortie)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(garmin_dataloader1_train)

            # Model evaluation
            model.eval()
            with torch.no_grad():
                eval_loss = 0
                for batch in garmin_dataloader1_eval:
                    sortie = model(batch)
                    loss  = loss_fn(batch, sortie)
                    eval_loss += loss.item()
            eval_loss = eval_loss / len(garmin_dataloader1_eval)

            print(f"eval loss -> {eval_loss}, and train loss -> {train_loss}")

            model.train()

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("eval_loss", eval_loss, step=epoch)

        mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path="model_state_dict")