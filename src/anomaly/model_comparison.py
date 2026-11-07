"""
model_comparison.py -- compare v1 and v2 of the anomaly detection models.

Goal: find out whether retraining on more data (v2) improved the Isolation Forest
and the autoencoder, using the same validation days for every model.

How: the v2 validation days are unseen by v1 and v2. They are converted to v1's
normalization before scoring v1. We compare reconstruction losses, rank stability
between versions (Spearman) and agreement between the two model families.
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from src.anomaly.model_autoencoder import Autoencoder, GarminDataset

COMPARISON_RESULT_PATH = "data/comparison_results.csv"

RUN_ID_IF_V1 = "556c363237564704aea67f2ee3bffa06"
RUN_ID_IF_V2 = "c6e17ccc7c764efa823e3f54370c606a"
RUN_ID_AE_V1 = "1b87381d170a4d1ea7fbee74358b19c3"


def load_ae(run_id):
    """Rebuild an Autoencoder from the state_dict saved in the MLflow run"""
    model = Autoencoder()
    model.load_state_dict(mlflow.pytorch.load_state_dict(f"runs:/{run_id}/model_state_dict"))
    model.eval()
    return model


def score_ae(model, array):
    """One reconstruction error per day: mean over time and channels (dims 1 and 2), dim 0 = day"""
    loss_fn = torch.nn.MSELoss(reduction="none") # "reduction=None" so that the loss let me do the
                                                # mean, but only on dim 1 (time) and 2 (feature) while
                                                # dim 3 (day) separated because we want a score for
                                                # each day
    loader = DataLoader(GarminDataset(array), batch_size=16)
    scores = []
    with torch.no_grad():
        for batch in loader:
            loss = loss_fn(model(batch), batch)
            scores.extend(loss.mean(dim=(1, 2)).tolist())
    return np.array(scores)


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # mlflow ui --backend-store-uri sqlite:///mlflow.db
    mlflow.set_experiment("models_comparison")

    data_v1 = np.load("data/dataset_normalized_v1.npz")
    data_v2 = np.load("data/dataset_normalized_v2.npz")
    print(data_v1.files)
    print(data_v2.files)

    dates_val = data_v2["dates_val"]
    # the v2 validation days must be unseen by v1
    assert dates_val[0] > data_v1["dates_train"][-1], "v2 val overlaps v1 train"

    # are the tabular features raw or z-scored?
    print("features_val v2: mean", data_v2["features_val"].mean(), "std", data_v2["features_val"].std())

    val_v2 = data_v2["val"]
    raw = val_v2 * data_v2["std"] + data_v2["mean"]  # back to real units
    val_v1 = ((raw - data_v1["mean"]) / data_v1["std"]).astype(np.float32)

    print(val_v1.shape, val_v1.dtype, val_v2.shape)
    print("mean per channel, v2 norm:", val_v2.mean(axis=(0, 1)))
    print("mean per channel, v1 norm:", val_v1.mean(axis=(0, 1)))

    ae_v2_ids = mlflow.search_runs(experiment_names=["autoencoder"], filter_string="tags.version = 'v2'")["run_id"].tolist()
    print("AE v2 runs:", len(ae_v2_ids))
    print("test score:", score_ae(load_ae(ae_v2_ids[0]), val_v2).mean())

    # AE v2: one score per day for each of the 5 seeds
    scores_ae_v2_seeds = np.array([score_ae(load_ae(run_id), val_v2) for run_id in ae_v2_ids])
    losses_v2 = scores_ae_v2_seeds.mean(axis=1)   # one val loss per seed
    score_ae_v2 = scores_ae_v2_seeds.mean(axis=0) # average over seeds, one value per day
    print("val loss per seed (v2):", losses_v2)

    # AE v1, scored on the same days in v1's normalization
    score_ae_v1 = score_ae(load_ae(RUN_ID_AE_V1), val_v1)
    print("val loss (v1):", score_ae_v1.mean())

    # Isolation Forest (lower score_samples = more abnormal)
    features_val = data_v2["features_val"]
    score_if_v1 = mlflow.sklearn.load_model(f"runs:/{RUN_ID_IF_V1}/model").score_samples(features_val)
    score_if_v2 = mlflow.sklearn.load_model(f"runs:/{RUN_ID_IF_V2}/model").score_samples(features_val)


    # Higher = more abnormal for every score (the IF score is flipped)
    abnormal_if_v1, abnormal_if_v2 = -score_if_v1, -score_if_v2

    def rho(a, b):
        return spearmanr(a, b)[0]

    metrics = {
        "val_loss_ae_v1": score_ae_v1.mean(),
        "val_loss_ae_v2_mean": losses_v2.mean(),
        "val_loss_ae_v2_std": losses_v2.std(ddof=1),
        "spearman_if_v1_vs_v2": rho(abnormal_if_v1, abnormal_if_v2),
        "spearman_ae_v1_vs_v2": rho(score_ae_v1, score_ae_v2),
        "spearman_if_vs_ae_v1": rho(abnormal_if_v1, score_ae_v1),
        "spearman_if_vs_ae_v2": rho(abnormal_if_v2, score_ae_v2),
    }

    with mlflow.start_run():
        mlflow.log_params({
            "n_val_days": len(dates_val),
            "val_start": str(dates_val[0]),
            "n_ae_v2_seeds": len(ae_v2_ids),
        })
        mlflow.log_metrics({name: float(value) for name, value in metrics.items()})

    df = pd.DataFrame({
        "date": dates_val,
        "if_v1": score_if_v1, "if_v2": score_if_v2,
        "ae_v1": score_ae_v1, "ae_v2_mean": score_ae_v2,
    })
    df.to_csv(COMPARISON_RESULT_PATH, index=False)

    for name, value in metrics.items():
        print(f"{name:<24} {value:.3f}")
    print("Top 5 abnormal days, IF v1:", df.nsmallest(5, "if_v1")["date"].tolist())
    print("Top 5 abnormal days, IF v2:", df.nsmallest(5, "if_v2")["date"].tolist())
    print("Top 5 abnormal days, AE v1:", df.nlargest(5, "ae_v1")["date"].tolist())
    print("Top 5 abnormal days, AE v2:", df.nlargest(5, "ae_v2_mean")["date"].tolist())