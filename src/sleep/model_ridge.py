"""
model_ridge.py -- train and evaluate a Ridge regression on sleep_score

Goal: predict the sleep score of the night D->D+1 from day-D features
(normalized)

How: load the normalized sleep dataset -> fit Ridge on train -> MAE on val
-> log params/metrics/model in MLflow, with the baseline MAE alongside.
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

TARGET = "sleep_score"
ALPHAS = [1, 10, 100, 300, 1000, 3000]
N_SPLITS = 5
ACCUMULATION_MARKERS = ("_ema", "_acwr")  # features built from several past days


def run_experiment(run_name, X_train, X_val, y_train, y_val, feature_names, baselines):
    """Fit RidgeCV (alpha chosen on train only), evaluate on val, log to MLflow."""
    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    with mlflow.start_run(run_name=run_name):
        # 1. Choose alpha by time-series CV on train only
        alpha_search = RidgeCV(
            alphas=ALPHAS, cv=cv, scoring="neg_mean_absolute_error"
        ).fit(X_train, y_train)
        best_alpha = float(alpha_search.alpha_)

        # 2. Plain Ridge with that alpha (cleaner to log than the RidgeCV object)
        model = Ridge(alpha=best_alpha).fit(X_train, y_train)

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
        mlflow.log_param("alpha_selected", float(best_alpha))
        mlflow.log_param("alpha_grid", ALPHAS)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.sklearn.log_model(model, name="model")

        print(f"\n=== {run_name} ({X_train.shape[1]} features) ===")
        print(f"alpha selected by CV: {best_alpha}")
        print(f"Train  MAE {metrics['train_mae']:.2f}  R2 {metrics['train_r2']:.3f}")
        print(f"Val    MAE {metrics['val_mae']:.2f}  R2 {metrics['val_r2']:.3f}")

        # Top coefficients (features are z-scored, so magnitudes are comparable)
        order = np.argsort(-np.abs(model.coef_))[:8]
        print("Top coefficients:")
        for i in order:
            print(f"  {feature_names[i]:<35} {model.coef_[i]:+.3f}")

    return metrics


if __name__ == "__main__":
    data_loaded = np.load("data/sleep_dataset_normalized.npz")

    causes_train_norm = data_loaded["causes_train_norm"]
    causes_val_norm = data_loaded["causes_val_norm"]
    effects_train = data_loaded["effects_train"]
    effects_val = data_loaded["effects_val"]
    feature_names = list(data_loaded["cause_metrics"])
    effect_metrics = list(data_loaded["effect_metrics"])

    target_idx = effect_metrics.index(TARGET)
    y_train = effects_train[:, target_idx]
    y_val = effects_val[:, target_idx]

    # Baselines: constant prediction from the train set
    baselines = {
        "baseline_mae_mean": mean_absolute_error(y_val, np.full_like(y_val, y_train.mean())),
        "baseline_mae_median": mean_absolute_error(y_val, np.full_like(y_val, np.median(y_train))),
    }
    print(f"Baseline MAE (mean):   {baselines['baseline_mae_mean']:.2f}")
    print(f"Baseline MAE (median): {baselines['baseline_mae_median']:.2f}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # mlflow ui --backend-store-uri sqlite:///mlflow.db
    mlflow.set_experiment("sleep_score_prediction")

    # Experiment 1: all features (base + EMA/ACWR)
    run_experiment("ridge_all_features", causes_train_norm, causes_val_norm,
                   y_train, y_val, feature_names, baselines)

    # Experiment 2: base features only (ablation of the accumulation features)
    base_idx = [i for i, name in enumerate(feature_names)
                if not any(marker in name for marker in ACCUMULATION_MARKERS)]
    base_names = [feature_names[i] for i in base_idx]
    run_experiment("ridge_no_ema", causes_train_norm[:, base_idx], causes_val_norm[:, base_idx],
                   y_train, y_val, base_names, baselines)