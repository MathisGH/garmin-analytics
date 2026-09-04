"""
dataset.py -- prepare the normalized train/val dataset for modeling.

Goal: turn the raw (n_days, 288, 4) array from features.py into a
chronologically split, per-channel normalized dataset ready to feed into
a PyTorch Dataset later, saved to disk so it doesn't need rebuilding.

How: build_dataset() from features.py -> chronological 85/15 train/val
split by index -> per-channel z-score normalization (mean/std fit on
train only) -> everything saved to data/dataset_normalized.npz.
"""

from src.features import build_dataset, build_tabular_features
from datetime import date
import numpy as np
from pathlib import Path

TRAIN_VAL_SPLIT_RATIO = 0.85
ARRAYS_PATH = Path("data")
DB_PATH = "data/garmin_data.db"

if __name__ == "__main__":
    start_date = date(2026, 3, 1)
    end_date = date.today()
    final, all_dates = build_dataset(start_date, end_date, DB_PATH)

    final_features = build_tabular_features(final) # For the scikit-learn model

    # Train/test split
    cutoff = int(TRAIN_VAL_SPLIT_RATIO*len(all_dates))

    final_train = final[:cutoff]
    final_features_train = final_features[:cutoff]
    dates_train = all_dates[:cutoff]

    final_val = final[cutoff:]
    final_features_val = final_features[cutoff:]
    dates_val = all_dates[cutoff:]

    # Normalization
    mean = final_train.mean(axis=(0, 1)) # I "deleted" 2 dimensions (axis=(0,1)) at once in order to get the mean for each column
    std = final_train.std(axis=(0, 1))

    final_train_norm = (final_train - mean) / std
    final_val_norm = (final_val - mean) / std

    dates_train_array = np.array(dates_train) # In order to save it in the npz file
    dates_val_array = np.array(dates_val)


    np.savez(file = ARRAYS_PATH / "dataset_normalized.npz", train = final_train_norm, val = final_val_norm,
            dates_train = dates_train_array, dates_val = dates_val_array, mean = mean, std = std, 
            features_train = final_features_train, features_val = final_features_val)