"""
model_random_forest.py -- train and evaluate a Random Forest on sleep_score

Goal: check whether non-linear effects / interactions between day-D features
predict the night's sleep score better than the Ridge regression.

How: load the sleep dataset (raw features, no normalization needed for trees)
-> pick min_samples_leaf / max_features by time-series CV on train only
-> evaluate once on val -> log metrics and feature importances in MLflow.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

TARGET = "sleep_score"
N_ESTIMATORS = 300
RANDOM_STATE = 15
N_SPLITS = 5
PARAM_GRID = {
    "min_samples_leaf": [5, 10, 20],
    "max_features": [0.3, 0.6, 1.0],
}
ACCUMULATION_MARKERS = ("_ema", "_acwr")


def run_experiment(run_name, X_train, X_val, y_train, y_val, feature_names, baselines):
    with mlflow.start_run(run_name=run_name):
        search = GridSearchCV(
            RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1),
            param_grid=PARAM_GRID,
            cv=TimeSeriesSplit(n_splits=N_SPLITS),
            scoring="neg_mean_absolute_error",
        ).fit(X_train, y_train)

        model = search.best_estimator_
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        metrics = {
            "train_mae": mean_absolute_error(y_train, train_pred),
            "train_r2": r2_score(y_train, train_pred),
            "val_mae": mean_absolute_error(y_val, val_pred),
            "val_r2": r2_score(y_val, val_pred),
            **baselines,
        }

        mlflow.log_param("target", TARGET)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        for param, value in search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.sklearn.log_model(model, name="model")

        print(f"\n=== {run_name} ({X_train.shape[1]} features) ===")
        print(f"best params (CV): {search.best_params_}")
        print(f"Train  MAE {metrics['train_mae']:.2f}  R2 {metrics['train_r2']:.3f}")
        print(f"Val    MAE {metrics['val_mae']:.2f}  R2 {metrics['val_r2']:.3f}")

        order = np.argsort(-model.feature_importances_)[:10]
        print("Top feature importances:")
        for i in order:
            print(f"  {feature_names[i]:<35} {model.feature_importances_[i]:.3f}")

    return metrics


if __name__ == "__main__":
    data_loaded = np.load("data/sleep_dataset_normalized.npz")

    causes_train = data_loaded["causes_train"]
    causes_val = data_loaded["causes_val"]
    effects_train = data_loaded["effects_train"]
    effects_val = data_loaded["effects_val"]
    feature_names = list(data_loaded["cause_metrics"])
    effect_metrics = list(data_loaded["effect_metrics"])

    target_idx = effect_metrics.index(TARGET)
    y_train = effects_train[:, target_idx]
    y_val = effects_val[:, target_idx]

    baselines = {
        "baseline_mae_mean": mean_absolute_error(y_val, np.full_like(y_val, y_train.mean())),
        "baseline_mae_median": mean_absolute_error(y_val, np.full_like(y_val, np.median(y_train))),
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sleep_score_prediction")

    run_experiment("rf_all_features", causes_train, causes_val,
                   y_train, y_val, feature_names, baselines)

    base_idx = [i for i, name in enumerate(feature_names)
                if not any(marker in name for marker in ACCUMULATION_MARKERS)]
    base_names = [feature_names[i] for i in base_idx]
    run_experiment("rf_no_ema", causes_train[:, base_idx], causes_val[:, base_idx],
                   y_train, y_val, base_names, baselines)