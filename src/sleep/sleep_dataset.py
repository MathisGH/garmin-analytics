"""
sleep_dataset.py -- chronological train/val split + normalization for the sleep dataset

Goal: split build_sleep_dataset() output 85/15 by date, z-score normalize
causes (fit on train only), save everything to data/sleep_dataset_normalized.npz
"""

from src.sleep.sleep_features import build_sleep_dataset, cause_metrics, effect_metrics
import numpy as np
from pathlib import Path

TRAIN_VAL_SPLIT_RATIO = 0.85
ARRAYS_PATH = Path("data")
DB_PATH = "data/garmin_data.db"

if __name__ == "__main__":
    dataset = build_sleep_dataset(DB_PATH)

    cutoff = int(TRAIN_VAL_SPLIT_RATIO * len(dataset))

    train = dataset.iloc[:cutoff]
    val = dataset.iloc[cutoff:]

    causes_train = train[cause_metrics].values
    causes_val = val[cause_metrics].values

    effects_train = train[effect_metrics].values
    effects_val = val[effect_metrics].values

    dates_train = train["date"].astype(str).values
    dates_val = val["date"].astype(str).values

    # Normalization (causes only, fit on train)
    mean = causes_train.mean(axis=0)
    std = causes_train.std(axis=0)

    causes_train_norm = (causes_train - mean) / std
    causes_val_norm = (causes_val - mean) / std

    np.savez(
        file=ARRAYS_PATH / "sleep_dataset_normalized.npz",
        causes_train=causes_train, causes_val=causes_val,
        causes_train_norm=causes_train_norm, causes_val_norm=causes_val_norm,
        effects_train=effects_train, effects_val=effects_val,
        dates_train=dates_train, dates_val=dates_val,
        mean=mean, std=std,
        cause_metrics=np.array(cause_metrics), effect_metrics=np.array(effect_metrics),
    )

    print(f"Train: {len(train)} days, Val: {len(val)} days")