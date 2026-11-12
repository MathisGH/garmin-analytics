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