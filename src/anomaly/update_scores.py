"""
update_scores.py -- update the JSON score file (locally and in the S3 bucket)

Goal: refresh scores_precomputed.json with all currently
extracted days, using the frozen models and normalization stats (no
retraining, no train/val split)

How: 
"""

from datetime import date
import time
import numpy as np
import mlflow
import torch
from torch.utils.data import DataLoader
import json

import boto3

from src.features import build_dataset, build_tabular_features
from src.anomaly.model_autoencoder import Autoencoder, GarminDataset
from src.anomaly.precompute_scores import score_day, RUN_ID_IF, RUN_ID_AE, OUTPUT_PATH

DB_PATH = "data/garmin_data.db"
mlflow.set_tracking_uri("sqlite:///mlflow.db")

S3_BUCKET = "garmin-anomaly-mathisdurand-2026"
S3_KEY = "scores_precomputed.json"
LAMBDA_FUNCTION_NAME = "garmin-anomaly-api"

def republish():
    s3_client = boto3.client("s3")
    s3_client.upload_file(OUTPUT_PATH, S3_BUCKET, S3_KEY)
    print(f"Uploaded {OUTPUT_PATH} to s3://{S3_BUCKET}/{S3_KEY}")

    lambda_client = boto3.client("lambda")
    lambda_client.update_function_configuration(
        FunctionName=LAMBDA_FUNCTION_NAME,
        Environment={
            "Variables": {
                "LAST_REFRESH": str(int(time.time()))
            }
        }
    )
    print("Triggered Lambda cold start via env var update")


if __name__ == "__main__":
    start_date = date(2026, 3, 1)
    end_date = date.today()

    # 1. Rebuild raw (real-unit) matrix + tabular features for ALL days
    raw_full, dates_full = build_dataset(start_date, end_date, DB_PATH)
    features_full = build_tabular_features(raw_full)

    # 2. Load FROZEN stats (not recomputed here)
    data_loaded = np.load("data/dataset_normalized.npz")
    mean = data_loaded["mean"]
    std = data_loaded["std"]
    mean_feat = data_loaded["mean_feat"]
    std_feat = data_loaded["std_feat"]

    # 3. Load frozen models
    if_model = mlflow.sklearn.load_model(f"runs:/{RUN_ID_IF}/model")
    ae_module = Autoencoder()
    state_dict = mlflow.pytorch.load_state_dict(f"runs:/{RUN_ID_AE}/model_state_dict")
    ae_module.load_state_dict(state_dict)
    ae_module.eval()

    # 4. IF score (unnormalized features, same as training)
    score_if_full = if_model.score_samples(features_full)

    # 5. AE score (raw signal normalized with frozen mean/std)
    raw_full_norm = (raw_full - mean) / std
    garmin_dataset = GarminDataset(raw_full_norm)
    garmin_dataloader = DataLoader(dataset=garmin_dataset, batch_size=16, shuffle=False)
    loss_fn = torch.nn.MSELoss(reduction='none')

    with torch.no_grad():
        scores_ae_global, scores_ae_channel = [], []
        for batch in garmin_dataloader:
            sortie = ae_module(batch)
            loss = loss_fn(batch, sortie)
            scores_ae_global.extend(loss.mean(dim=(1, 2)).tolist())
            scores_ae_channel.extend(loss.mean(dim=1).tolist())
        scores_ae_global = np.array(scores_ae_global)
        scores_ae_channel = np.array(scores_ae_channel)

    # 6. Z-scores with frozen tabular stats
    z_scores_full = (features_full - mean_feat) / std_feat

    # 7. raw_full is ALREADY real-unit -> no denormalization needed
    raw_real_full = raw_full

    # 8. Build + save results (reuses score_day from precompute_score.py)
    results = {}
    for i, day_date in enumerate(dates_full):
        results[str(day_date)] = score_day(
            i, day_date, score_if_full, scores_ae_global,
            scores_ae_channel, z_scores_full, raw_real_full
        )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} days to {OUTPUT_PATH}")
    republish()