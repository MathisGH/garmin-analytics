"""
test_database.py -- tests for src/database.py

Goal: check that the SQLite helpers create the schema, ignore duplicates
and update rows correctly.

How: each test works on a temporary database (pytest's tmp_path fixture),
never on data/garmin_data.db. tmp_path is a fresh folder deleted after the run.
"""

import sqlite3
import pytest
from src import database


def test_create_schema_creates_both_tables(db_path):
    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    connection.close()
    table_names = {row[0] for row in rows}
    assert {"daily_summary", "timeseries_data"} <= table_names


def test_insert_timeseries_ignores_duplicates(db_path):
    database.ensure_daily_row(db_path, "2026-01-01")
    points = [{"ts": 1000, "val": 60.0}, {"ts": 2000, "val": 62.0}]

    # Same points inserted twice: the UNIQUE constraint must prevent duplicates
    database.insert_timeseries(db_path, "2026-01-01", "heart_rate", points, "ts", "val")
    database.insert_timeseries(db_path, "2026-01-01", "heart_rate", points, "ts", "val")

    connection = sqlite3.connect(db_path)
    count = connection.execute("SELECT COUNT(*) FROM timeseries_data").fetchone()[0]
    connection.close()
    assert count == 2


def test_update_daily_summary_and_extracted_flag(db_path):
    database.ensure_daily_row(db_path, "2026-01-01")
    assert database.is_day_already_extracted(db_path, "2026-01-01") is False

    database.update_daily_summary(db_path, "2026-01-01", {"steps": 1000, "extracted": 1})

    assert database.is_day_already_extracted(db_path, "2026-01-01") is True
    assert database.is_day_already_extracted(db_path, "2026-01-02") is False