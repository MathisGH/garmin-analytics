"""
model_comparison.py -- compare every model on the anomaly detection task.

Goal: ...

How: ...
"""

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model_autoencoder import Autoencoder, GarminDataset

mlflow.set_tracking_uri("sqlite:///mlflow.db") # mlflow ui --backend-store-uri sqlite:///mlflow.db
mlflow.set_experiment("models_comparison")

run_id_if = "39e9923ce74944019526cfe5983040d3"
run_id_ae = "301696e089894314b5f8312a798b9e4b"

if_model = mlflow.sklearn.load_model(f"runs:/{run_id_if}/model")

ae_module = Autoencoder()
state_dict = mlflow.pytorch.load_state_dict(f"runs:/{run_id_ae}/model_state_dict")
ae_module.load_state_dict(state_dict)
ae_module.eval()


loss_fn = torch.nn.MSELoss(reduction='none') # "reduction=None" so that the loss let me do the
                                           # mean, but only on dim 1 (time) and 2 (feature) while
                                           # dim 3 (day) separated because we want a score for
                                           # each day

    
data_loaded = np.load("data/dataset_normalized.npz")
eval_array = data_loaded["val"]
features_val = data_loaded["features_val"]
dates_val = data_loaded["dates_val"]

garmin_dataset_eval = GarminDataset(eval_array)
garmin_dataloader_eval = DataLoader(dataset=garmin_dataset_eval, batch_size=16)


with mlflow.start_run():
    # Isolation Forest model:
    score_if = if_model.score_samples(features_val)

    # Autoencoder model
    ae_module.eval()
    with torch.no_grad():
        scores_ae = []
        for batch in garmin_dataloader_eval:
            sortie = ae_module(batch)
            loss  = loss_fn(batch, sortie)
            loss_per_day = loss.mean(dim=(1, 2)) # mean on time+features, keep 1 value/day
            scores_ae.extend(loss_per_day.tolist())
        scores_ae = np.array(scores_ae)

    df = pd.DataFrame({
    "date": dates_val,
    "score_if": score_if,      # higher = normal for IF
    "score_ae": scores_ae,     # lower = normal for AE
    })

    # need to use ranks to compare these two different scores
    df["rank_if"] = df["score_if"].rank(ascending=True)   # rank 1 = normal
    df["rank_ae"] = df["score_ae"].rank(ascending=False)  # rank 1 = normal

    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(df["rank_if"], df["rank_ae"])

    mlflow.log_metric("spearman_correlation", correlation)