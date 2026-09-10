"""
precompute_score.py -- ... .

Goal: Rather than computing the score at every API call, store each anomaly score (AE and IF) into a JSON file
     containing every metrics (the raw one and the normalized one) for each day

How: ...
"""

import numpy as np
import pandas as pd
import mlflow
import torch
from torch.utils.data import DataLoader
from src.model_autoencoder import Autoencoder, GarminDataset
import json



mlflow.set_tracking_uri("sqlite:///mlflow.db") # mlflow ui --backend-store-uri sqlite:///mlflow.db


RUN_ID_IF = "556c363237564704aea67f2ee3bffa06"
RUN_ID_AE = "1b87381d170a4d1ea7fbee74358b19c3"
OUTPUT_PATH = "data/scores_precomputed.json"

CHANNEL_NAMES = ['body_battery', 'heart_rate', 'respiration', 'stress']   # exact order of each 4 channels in raw_full
FEATURE_NAMES = ['mean_body_battery', 'mean_heart_rate', 'mean_respiration', 'mean_stress',
                 'std_body_battery', 'std_heart_rate', 'std_respiration', 'std_stress',
                 'min_body_battery', 'min_heart_rate', 'min_respiration', 'min_stress',
                 'max_body_battery', 'max_heart_rate', 'max_respiration', 'max_stress',]   # exact order of each 16 columns in features_full

if __name__ == "__main__":
    # Data
    data_loaded = np.load("data/dataset_normalized.npz")

    features_train = data_loaded["features_train"]
    features_val = data_loaded["features_val"]
    features_full = np.concatenate([features_train, features_val])

    raw_train = data_loaded["train"]
    raw_val = data_loaded["val"]
    raw_full = np.concatenate([raw_train, raw_val])

    dates_val = data_loaded["dates_val"]
    dates_train = data_loaded["dates_train"]
    dates_full = np.concatenate([dates_train, dates_val])

    mean = data_loaded["mean"]
    std = data_loaded["std"]

    # Models

    if_model = mlflow.sklearn.load_model(f"runs:/{RUN_ID_IF}/model")

    ae_module = Autoencoder()
    state_dict = mlflow.pytorch.load_state_dict(f"runs:/{RUN_ID_AE}/model_state_dict")
    ae_module.load_state_dict(state_dict)
    ae_module.eval()

    # Scores
    score_if_full = if_model.score_samples(features_full)

    garmin_dataset = GarminDataset(raw_full)
    garmin_dataloader = DataLoader(dataset=garmin_dataset, batch_size=16, shuffle=False)
    loss_fn = torch.nn.MSELoss(reduction='none')

    with torch.no_grad():
            scores_ae_global = []
            scores_ae_channel = []
            for batch in garmin_dataloader:
                sortie = ae_module(batch)
                loss  = loss_fn(batch, sortie)
                loss_per_day = loss.mean(dim=(1, 2)) # mean on time+features, keep 1 value/day -> the average reconstruction loss per day
                loss_per_day_bis = loss.mean(dim=1)
                scores_ae_global.extend(loss_per_day.tolist())
                scores_ae_channel.extend(loss_per_day_bis.tolist())
            scores_ae_global = np.array(scores_ae_global)
            scores_ae_channel = np.array(scores_ae_channel)

    mean_train_feat = features_train.mean(axis=0)  # shape (16,)
    std_train_feat = features_train.std(axis=0)    # shape (16,)
    z_scores_full = (features_full - mean_train_feat) / std_train_feat  # shape (146, 16)

    raw_real_full = raw_full * std + mean  # denormalizing (more interpretable), shape (146, 288, 4)

    results = {}
    for i, day_date in enumerate(dates_full):
        results[str(day_date)] = {
            "score_if": float(score_if_full[i]),
            "score_ae_global": float(scores_ae_global[i]),
            "score_ae_by_channel": dict(zip(CHANNEL_NAMES, scores_ae_channel[i].tolist())),
            "z_scores": dict(zip(FEATURE_NAMES, z_scores_full[i].tolist())),
            "raw_series": {
                ch: raw_real_full[i, :, c].tolist()
                for c, ch in enumerate(CHANNEL_NAMES)
            },
        }


    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} days to {OUTPUT_PATH}")