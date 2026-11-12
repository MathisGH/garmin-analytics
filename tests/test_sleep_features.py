"""
test_sleep_features.py -- tests for src/sleep/sleep_features.py

Goal: guarantee that the day-D causes are aligned with the night D+1 sleep,
and that nothing from the night (or later) leaks into its own features.

How: synthetic database from conftest.py (value of day i = 100 + i).
"""

import pandas as pd
from src import database
from src.sleep.sleep_features import build_sleep_dataset, cause_metrics
import sqlite3

EXPECTED_ROWS = 39  # N_DAYS (40, see conftest.py) minus the first day

def test_first_day_is_dropped(sleep_db_path):
    result = build_sleep_dataset(sleep_db_path)
    assert len(result) == EXPECTED_ROWS


def get_row(dataset, day):
    """One-row DataFrame for the given date (kept as a DataFrame on purpose)"""
    return dataset[dataset["date"] == pd.Timestamp(day)]


def test_causes_come_from_previous_day(sleep_db_path):
    result = build_sleep_dataset(sleep_db_path)
    row = get_row(result, "2026-01-10").iloc[0]

    connection = sqlite3.connect(sleep_db_path)
    steps_previous_day = connection.execute(
        "SELECT steps FROM daily_summary WHERE date = ?", ("2026-01-09",)
    ).fetchone()[0]
    score_same_day = connection.execute(
        "SELECT sleep_score FROM daily_summary WHERE date = ?", ("2026-01-10",)
    ).fetchone()[0]
    connection.close()

    assert row["steps"] == steps_previous_day
    assert row["sleep_score"] == score_same_day


def test_changing_a_day_does_not_alter_its_own_row(sleep_db_path):
    before = build_sleep_dataset(sleep_db_path)
    database.update_daily_summary(sleep_db_path, "2026-01-20", {"steps": 400, "calories": 400})
    after = build_sleep_dataset(sleep_db_path)

    # Row of the 20th: must be untouched (its causes come from the 19th)
    pd.testing.assert_frame_equal(
        get_row(before, "2026-01-20")[cause_metrics],
        get_row(after, "2026-01-20")[cause_metrics],
    )
    # Row of the 21st: its causes come from the 20th, so it must change
    assert get_row(before, "2026-01-21")["steps"].iloc[0] != get_row(after, "2026-01-21")["steps"].iloc[0]