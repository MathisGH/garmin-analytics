"""
test_anomaly_features.py -- tests for src/anomaly/anomaly_features.py

Goal: freeze the shape, column order and quality filter of the daily matrices.

How: synthetic db from conftest.py, with one full day, one sparse day and one missing day.
"""

from datetime import date

import numpy as np

from src.anomaly.anomaly_features import (
    build_daily_matrix,
    build_dataset,
    build_tabular_features,
)
from tests.conftest import ANOMALY_FULL_DAY, ANOMALY_METRIC_VALUES, ANOMALY_SPARSE_DAY


def test_build_daily_matrix_full_day(anomaly_connection):
    df = build_daily_matrix(ANOMALY_FULL_DAY, anomaly_connection)

    assert df.shape == (288, 4)
    assert list(df.columns) == ["body_battery", "heart_rate", "respiration", "stress"]
    assert df.isna().sum().sum() == 0
    for metric, value in ANOMALY_METRIC_VALUES.items():
        np.testing.assert_allclose(df[metric], value)


def test_build_daily_matrix_rejects_incomplete_or_missing_day(anomaly_connection):
    assert build_daily_matrix(ANOMALY_SPARSE_DAY, anomaly_connection) is None  # too many gaps
    assert build_daily_matrix("2026-01-05", anomaly_connection) is None        # no data at all


def test_build_dataset_keeps_only_valid_days(anomaly_db_path):
    # Jan 1 = full day, Jan 2 = sparse day, Jan 3 = no data; end date is excluded
    final, dates = build_dataset(date(2026, 1, 1), date(2026, 1, 4), anomaly_db_path)

    assert final.shape == (1, 288, 4)
    assert final.dtype == np.float32
    assert dates == ["2026-01-01"]


def test_build_tabular_features_layout():
    array = np.ones((2, 288, 4)) * np.array([1, 2, 3, 4])  # constant channels
    features = build_tabular_features(array)

    # order: 4 means, 4 stds, 4 mins, 4 maxs
    expected_row = [1, 2, 3, 4,  0, 0, 0, 0,  1, 2, 3, 4,  1, 2, 3, 4]
    assert features.shape == (2, 16)
    np.testing.assert_allclose(features, np.tile(expected_row, (2, 1)), atol=1e-9)