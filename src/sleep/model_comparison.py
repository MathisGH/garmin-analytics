"""
model_comparison.py -- compare baseline / Ridge / Random Forest on all sleep targets.

Goal: measure how much each model improves over
a constant "predict the median" baseline, for sleep_score, sleep_time_seconds
and the proportion of each sleep stage (stage seconds / total sleep time).

How: expanding-window time-series cross-validation over the whole dataset
(train + val concatenated, still chronological). Ridge is wrapped in a pipeline
(scaler + RidgeCV) so normalization and alpha selection happen inside each fold
only, without leakage. Results are saved to CSV and logged in MLflow.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import mlflow

N_SPLITS = 5
INNER_SPLITS = 3
RIDGE_ALPHAS = [1, 10, 100, 300, 1000, 3000]
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 20, "max_features": 0.6,
             "random_state": 42, "n_jobs": -1}
ACCUMULATION_MARKERS = ("_ema", "_acwr")
STAGES = ["deep", "light", "rem", "awake"]
OUTPUT_PATH = Path("data/sleep_model_comparison.csv")


def make_ridge():
    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=RIDGE_ALPHAS, cv=TimeSeriesSplit(n_splits=INNER_SPLITS),
                scoring="neg_mean_absolute_error"),
    )


def build_models():
    """name -> (estimator, feature set to use: 'all' or 'base' = without EMA/ACWR)"""
    return {
        "baseline_median": (DummyRegressor(strategy="median"), "all"),
        "ridge": (make_ridge(), "all"),
        "ridge_no_ema": (make_ridge(), "base"),
        "random_forest": (RandomForestRegressor(**RF_PARAMS), "all"),
        "random_forest_no_ema": (RandomForestRegressor(**RF_PARAMS), "base"),
    }


def build_targets(effects, effect_metrics):
    """Targets: score, total duration, and stage proportions (isolate structure from duration)."""
    col = {name: effects[:, effect_metrics.index(name)] for name in effect_metrics}
    total = col["sleep_time_seconds"]
    targets = {"sleep_score": col["sleep_score"], "sleep_time_seconds": total}
    for stage in STAGES:
        targets[f"{stage}_sleep_ratio"] = col[f"{stage}_sleep_seconds"] / total
    return targets


if __name__ == "__main__":
    data_loaded = np.load("data/sleep_dataset_normalized.npz")

    # train rows come before val rows in time, so concatenation stays chronological
    X = np.concatenate([data_loaded["causes_train"], data_loaded["causes_val"]])
    effects = np.concatenate([data_loaded["effects_train"], data_loaded["effects_val"]])
    feature_names = list(data_loaded["cause_metrics"])
    effect_metrics = list(data_loaded["effect_metrics"])

    feature_sets = {
        "all": list(range(len(feature_names))),
        "base": [i for i, name in enumerate(feature_names)
                 if not any(marker in name for marker in ACCUMULATION_MARKERS)],
    }

    targets = build_targets(effects, effect_metrics)
    models = build_models()
    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    rows = []
    for target_name, y in targets.items():
        print(f"Evaluating target: {target_name}")
        fold_mae = {}
        for model_name, (estimator, feature_set) in models.items():
            scores = cross_val_score(
                estimator, X[:, feature_sets[feature_set]], y,
                cv=cv, scoring="neg_mean_absolute_error",
            )
            fold_mae[model_name] = -scores  # one MAE per fold

        baseline = fold_mae["baseline_median"]
        for model_name, maes in fold_mae.items():
            is_baseline = model_name == "baseline_median"
            rows.append({
                "target": target_name,
                "model": model_name,
                "mae_mean": maes.mean(),
                "mae_std": maes.std(),
                "gain_vs_baseline_pct": 0.0 if is_baseline
                    else 100 * (1 - maes.mean() / baseline.mean()),
                "folds_beating_baseline": "-" if is_baseline
                    else f"{int((maes < baseline).sum())}/{N_SPLITS}",
            })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_PATH, index=False)
    print("\n" + results.round(4).to_string(index=False))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sleep_model_comparison")
    with mlflow.start_run(run_name="cv_comparison"):
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("n_rows", len(X))
        mlflow.log_param("ridge_alphas", RIDGE_ALPHAS)
        mlflow.log_param("rf_params", RF_PARAMS)
        for row in rows:
            prefix = f"{row['target']}__{row['model']}"
            mlflow.log_metric(f"{prefix}__mae_mean", row["mae_mean"])
            mlflow.log_metric(f"{prefix}__gain_pct", row["gain_vs_baseline_pct"])
        mlflow.log_artifact(str(OUTPUT_PATH))