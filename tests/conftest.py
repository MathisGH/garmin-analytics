"""
conftest.py -- shared pytest fixtures

Goal: provide synthetic data to any test, so no test ever touches the real
Garmin database.

How: sleep_db_path builds a temporary SQLite database with N_DAYS days where
every column of day i equals 100 + i. The values strictly increase and stay
positive, and each value tells us which day it
comes from, which makes the assertions easy to write.
"""

from datetime import date, timedelta
import pytest
from src import database
import pandas as pd
import numpy as np
import sqlite3

N_DAYS = 40
START_DATE = date(2026, 1, 1)
VALUE_OFFSET = 100

# Every daily_summary column read by build_sleep_dataset (a missing one would
# become NaN, and dropna would then remove all rows)
SLEEP_COLUMNS = [
    "steps", "calories", "distance_meters", "avg_stress", "max_stress",
    "resting_hr", "min_hr", "max_hr", "body_battery_charged",
    "body_battery_drained", "body_battery_highest", "body_battery_lowest",
    "avg_respiration", "intensity_minutes_moderate",
    "intensity_minutes_vigorous", "vo2max", "training_load_weekly",
    "sleep_score", "sleep_time_seconds", "deep_sleep_seconds",
    "light_sleep_seconds", "rem_sleep_seconds", "awake_sleep_seconds",
]

@pytest.fixture
def db_path(tmp_path): # tmp_path is given by pytest (a temporary file deleted after the test)
    """A fresh database with the schema created, for each test"""
    path = str(tmp_path / "data" / "test.db")
    database.create_schema(path)
    return path


@pytest.fixture
def sleep_db_path(tmp_path):
    """Temporary database where the column values of day i equal 100 + i"""
    path = str(tmp_path / "data" / "sleep_test.db")
    database.create_schema(path)
    for i in range(N_DAYS):
        day = (START_DATE + timedelta(days=i)).isoformat()
        database.ensure_daily_row(path, day)
        fields = {column: VALUE_OFFSET + i for column in SLEEP_COLUMNS}
        database.update_daily_summary(path, day, fields)
    return path


@pytest.fixture
def date_value_df():
    """20 consecutive days with a 'value' column equal to the day index (0..19)"""
    dates = pd.date_range("2026-01-01", periods=20)
    return pd.DataFrame({"date": dates, "value": np.arange(20)})


ANOMALY_FULL_DAY = "2026-01-01"
ANOMALY_SPARSE_DAY = "2026-01-02"
ANOMALY_METRIC_VALUES = {
    "body_battery": 50.0, "heart_rate": 60.0, "respiration": 15.0, "stress": 30.0,
}
FIVE_MINUTES_MS = 300_000


def _make_points(day, n_points, value):
    """n_points Garmin-like points (one every 5 min from midnight, in ms)"""
    start_ms = pd.Timestamp(day).value // 10**6  # .value is in nanoseconds
    return [
        {"timestamp": start_ms + i * FIVE_MINUTES_MS, "value": value}
        for i in range(n_points)
    ]


@pytest.fixture
def anomaly_db_path(tmp_path):
    """Database with one complete day (288 points/metric) and one sparse day (50)"""
    path = str(tmp_path / "data" / "anomaly_test.db")
    database.create_schema(path)
    for day, n_points in ((ANOMALY_FULL_DAY, 288), (ANOMALY_SPARSE_DAY, 50)):
        database.ensure_daily_row(path, day)
        for metric, value in ANOMALY_METRIC_VALUES.items():
            points = _make_points(day, n_points, value)
            database.insert_timeseries(path, day, metric, points, "timestamp", "value")
    return path


@pytest.fixture
def anomaly_connection(anomaly_db_path):
    """Open connection on the anomaly test database, closed after the test"""
    connection = sqlite3.connect(anomaly_db_path)
    yield connection
    connection.close()