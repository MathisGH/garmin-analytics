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


def split_chronologically(dataset, ratio):
    """Split a date-sorted DataFrame: the first ratio of the rows for train, the rest for val"""
    cutoff = int(ratio * len(dataset))
    return dataset.iloc[:cutoff], dataset.iloc[cutoff:]


def normalize_causes(causes_train, causes_val):
    """Z-score both sets with the mean/std of the TRAIN set only"""
    mean = causes_train.mean(axis=0)
    std = causes_train.std(axis=0)
    return (causes_train - mean) / std, (causes_val - mean) / std, mean, std


if __name__ == "__main__":
    dataset = build_sleep_dataset(DB_PATH)

    train, val = split_chronologically(dataset=dataset, ratio=TRAIN_VAL_SPLIT_RATIO)

    causes_train = train[cause_metrics].values
    causes_val = val[cause_metrics].values

    effects_train = train[effect_metrics].values
    effects_val = val[effect_metrics].values

    dates_train = train["date"].astype(str).values
    dates_val = val["date"].astype(str).values

    # Normalization (causes only, fit on train)
    causes_train_norm, causes_val_norm, mean, std = normalize_causes(causes_train, causes_val)

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