"""
precompute_sleep.py -- build the JSON file used by the sleep section of the web app.

Goal: precompute everything the API serves (nights, per-night contributions, global
coefficients, model comparison) so the Lambda only has to read a static file.

How: fit one Ridge model (StandardScaler + RidgeCV, 16 base features, target
sleep_score) on all nights. For each night, the contribution of a feature is
coefficient x standardized value; contributions + intercept = prediction.

Warning: predictions are in-sample, so they look better than reality. The honest
performance estimate is the cross-validated table in data/sleep_model_comparison.csv.

Usage: python -m src.sleep.precompute_sleep
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path("data/sleep_dataset_normalized.npz")
COMPARISON_PATH = Path("data/sleep_model_comparison.csv")
OUTPUT_PATH = Path("data/sleep_precomputed.json")

STAGES = ["deep", "light", "rem", "awake"]


def round_dict(names, values):
    return {name: round(float(value), 4) for name, value in zip(names, values)}


def load_data():
    """Return raw features, feature names, effects (dict of arrays) and dates, in chronological order."""
    data = np.load(DATASET_PATH, allow_pickle=True)
    X = np.concatenate([data["causes_train"], data["causes_val"]]).astype(float)
    effects = np.concatenate([data["effects_train"], data["effects_val"]]).astype(float)
    dates = np.concatenate([data["dates_train"], data["dates_val"]]).astype(str)
    feature_names = [str(name) for name in data["cause_metrics"]]
    effects = {str(name): effects[:, i] for i, name in enumerate(data["effect_metrics"])}
    return X, feature_names, effects, dates


if __name__ == "__main__":
    X, feature_names, effects, dates = load_data()

    # Keep the 16 base features (the EMA / ACWR columns do not help in cross-validation)
    keep = [i for i, name in enumerate(feature_names) if "_ema" not in name and "_acwr" not in name]
    X = X[:, keep]
    feature_names = [feature_names[i] for i in keep]

    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=[1, 10, 100, 300, 1000, 3000], cv=TimeSeriesSplit(n_splits=3),
                scoring="neg_mean_absolute_error"),
    )
    model.fit(X, effects["sleep_score"])
    scaler, ridge = model[0], model[1]

    contributions = scaler.transform(X) * ridge.coef_
    predictions = model.predict(X)

    # A stage at 0 seconds means the watch did not classify it: those nights have no usable stages
    stages_valid = (effects["deep_sleep_seconds"] > 0) & (effects["rem_sleep_seconds"] > 0) \
        & (effects["awake_sleep_seconds"] > 0)

    nights = []
    for i, date in enumerate(dates):
        night = {
            "date": date,
            "sleep_score": round(float(effects["sleep_score"][i]), 4),
            "predicted_score": round(float(predictions[i]), 4),
            "sleep_time_seconds": round(float(effects["sleep_time_seconds"][i]), 4),
            "stages_valid": bool(stages_valid[i]),
            "features": round_dict(feature_names, X[i]),
            "contributions": round_dict(feature_names, contributions[i]),
        }
        for stage in STAGES:
            night[f"{stage}_sleep_seconds"] = round(float(effects[f"{stage}_sleep_seconds"][i]), 4)
        nights.append(night)

    coefficients = sorted(
        [{"feature": name, "coefficient": round(float(coef), 4)}
         for name, coef in zip(feature_names, ridge.coef_)],
        key=lambda row: abs(row["coefficient"]), reverse=True,
    )

    output = {
        "meta": {
            "n_nights": len(dates),
            "n_nights_valid_stages": int(stages_valid.sum()),
            "date_start": dates[0],
            "date_end": dates[-1],
            "alpha": float(ridge.alpha_),
            "intercept": round(float(ridge.intercept_), 4),
        },
        "coefficients": coefficients,
        "model_comparison": pd.read_csv(COMPARISON_PATH).round(4).to_dict(orient="records"),
        "nights": nights,
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")

    print(f"{len(dates)} nights ({dates[0]} -> {dates[-1]}), {int(stages_valid.sum())} with valid stages, "
          f"alpha {ridge.alpha_}")
    print(f"Saved {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")