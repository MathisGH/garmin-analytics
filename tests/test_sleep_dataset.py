"""
test_sleep_dataset.py -- tests for src/sleep/sleep_dataset.py

Goal: check the chronological split and the normalization

How: 
"""

import numpy as np

from src.sleep.sleep_dataset import normalize_causes, split_chronologically


def test_temporal_split_date_order(date_value_df):
    result_train, result_val = split_chronologically(date_value_df, 0.85)

    assert result_train["date"].iloc[-1] < result_val["date"].iloc[0]


def test_temporal_split_no_lost_data(date_value_df):
    result_train, result_val = split_chronologically(date_value_df, 0.85)

    assert len(result_train) == 17
    assert len(result_train) + len(result_val) == len(date_value_df)


def test_normalization_on_train_set():
    train = np.arange(0, 17, dtype=float).reshape(-1, 1)    # shape (17, 1)
    val = np.arange(17, 20, dtype=float).reshape(-1, 1) * 50 # To make sure the values are different

    train_norm, val_norm, mean, std = normalize_causes(train, val)

    np.testing.assert_allclose(train_norm.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(train_norm.std(axis=0), 1)
    np.testing.assert_allclose(mean, train.mean(axis=0), atol=1e-9)
    np.testing.assert_allclose(std, train.std(axis=0), atol=1e-9)
    assert abs(val_norm.mean()) > 1