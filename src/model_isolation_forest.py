"""
model_isolation_forest.py -- train and evaluate the baseline Isolation Forest.

Goal: ...

How: ...
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn

data_loaded = np.load("data/dataset_normalized.npz")

features_train = data_loaded["features_train"]
features_val = data_loaded["features_val"]
dates_val = data_loaded["dates_val"]

mlflow.set_tracking_uri("sqlite:///mlflow.db") # mlflow ui --backend-store-uri sqlite:///mlflow.db
mlflow.set_experiment("isolation_forest_baseline")

with mlflow.start_run():
    if_model = IsolationForest(n_estimators=100, contamination=0.1, random_state=15)
    if_model.fit(features_train)
    score = if_model.score_samples(features_val)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("contamination", 0.1)
    mlflow.log_param("random_state", 15)
    mlflow.log_metric("mean_val_score", score.mean())
    mlflow.sklearn.log_model(if_model, "model")

d = {"dates":dates_val, "score":score}
test = pd.DataFrame(d)
test.sort_values(by=["score"], ascending=False, inplace=True)
print(test.describe())

index_max = np.where(dates_val == "2026-07-09")
print(index_max[0][0])
print(dates_val.tolist().index("2026-07-09"))
print(features_val[index_max])

mean_train = features_train.mean(axis=0)
std_train = features_train.std(axis=0)

z_scores = (features_val[index_max[0][0]] - mean_train) / std_train

labels = [f"{stat}_{metric}" for stat in ["mean", "std", "min", "max"]
          for metric in ["body_battery", "heart_rate", "respiration", "stress"]]

for label, z in zip(labels, z_scores):
    print(f"{label}: {z:.2f}")