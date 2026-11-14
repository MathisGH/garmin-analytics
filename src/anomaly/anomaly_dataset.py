"""
anomaly_dataset.py -- prepare the normalized train/val dataset for anomaly detection.

Goal: turn the raw (n_days, 288, 4) array from anomaly_features.py into a
chronologically split, per-channel normalized dataset saved to disk.

How: build_dataset() -> split by index: the last 30 days are the validation set,
everything before is the train set -> per-channel z-score normalization (mean/std
fit on train only) -> saved to data/dataset_normalized_v2.npz.
(v1 used an 85/15 split and was saved as dataset_normalized_v1.npz.)
"""

from datetime import date
from pathlib import Path

import numpy as np

from src.anomaly.anomaly_features import build_dataset, build_tabular_features

# TRAIN_VAL_SPLIT_RATIO = 0.85
ARRAYS_PATH = Path("data")
DB_PATH = "data/garmin_data.db"

if __name__ == "__main__":
    start_date = date(2026, 3, 1)
    end_date = date.today()
    final, all_dates = build_dataset(start_date, end_date, DB_PATH)

    final_features = build_tabular_features(final) # For the scikit-learn model

    # Train/test split
    #cutoff = int(TRAIN_VAL_SPLIT_RATIO*len(all_dates))
    cutoff_v2 = len(final) - 30 # for the re-training I will only use the last 30 days as the validation set

    final_train = final[:cutoff_v2]
    final_features_train = final_features[:cutoff_v2]
    dates_train = all_dates[:cutoff_v2]

    final_val = final[cutoff_v2:]
    final_features_val = final_features[cutoff_v2:]
    dates_val = all_dates[cutoff_v2:]

    # Normalization
    mean = final_train.mean(axis=(0, 1)) # I "deleted" 2 dimensions (axis=(0,1)) at once in order to get the mean for each column
    std = final_train.std(axis=(0, 1))

    final_train_norm = (final_train - mean) / std
    final_val_norm = (final_val - mean) / std

    dates_train_array = np.array(dates_train) # In order to save it in the npz file
    dates_val_array = np.array(dates_val)


    np.savez(file = ARRAYS_PATH / "dataset_normalized_v2.npz", train = final_train_norm, val = final_val_norm,
            dates_train = dates_train_array, dates_val = dates_val_array, mean = mean, std = std, 
            features_train = final_features_train, features_val = final_features_val)