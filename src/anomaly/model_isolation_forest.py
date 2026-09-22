"""
model_isolation_forest.py -- train and evaluate the baseline Isolation Forest.

Goal: ...

How: ...
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn

N_ESTIMATORS = 100
CONTAMINATION = 0.1
RANDOM_STATE = 15

if __name__ == "__main__":
    data_loaded = np.load("data/dataset_normalized.npz")

    features_train = data_loaded["features_train"]
    features_val = data_loaded["features_val"]
    dates_val = data_loaded["dates_val"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db") # mlflow ui --backend-store-uri sqlite:///mlflow.db
    mlflow.set_experiment("isolation_forest_baseline")

    with mlflow.start_run():
        if_model = IsolationForest(n_estimators=N_ESTIMATORS, contamination=CONTAMINATION, random_state=RANDOM_STATE)
        if_model.fit(features_train)
        score = if_model.score_samples(features_val)

        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
        })
        mlflow.log_metric("mean_val_score", score.mean())
        mlflow.sklearn.log_model(if_model, "model")