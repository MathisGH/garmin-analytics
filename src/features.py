"""
features.py -- turn raw SQLite timeseries into clean per-day matrices.

Goal: for each date, produce a fixed-shape (288, 4) matrix (5-min grid x
4 metrics), handling missing data and dropping days that are too
incomplete, so downstream code always gets uniform daily "snapshots".

How: build_daily_matrix() pivots + resamples one day onto a fixed
calendar grid and interpolates gaps; build_dataset() loops this over a
date range and stacks the results into one (n_days, 288, 4) array.
"""

from datetime import timedelta, date
import pandas as pd
import sqlite3
import numpy as np


def build_daily_matrix(dates_str, connection):
    timeseries_data = pd.read_sql_query(
        "SELECT * FROM timeseries_data WHERE date = (?) ORDER BY timestamp",
        con = connection,
        params= (dates_str,)
    )

    if timeseries_data.empty:
        # print(f"Empty row for date: {dates_str}")
        return None
    
    timeseries_pivoted = timeseries_data.pivot_table(index="timestamp", columns="metric", values="value")
    
    # 1. Columns we need/expect
    expected_metrics = ['body_battery', 'heart_rate', 'respiration', 'stress']
    
    # 2. .reindex() in order to create NaN columns without getting errors
    df = timeseries_pivoted.reindex(columns=expected_metrics)
    
    df.index = pd.to_datetime(df.index, unit="ms")
    df_resampled = df.resample("5min").mean()

    # 4. In order to have the good shape for the grid (Exactly 288 steps from 00:00 to 23:55)
    day_start = pd.Timestamp(dates_str)
    fixed_grid = pd.date_range(start=day_start, periods=288, freq="5min")
    df_resampled = df_resampled.reindex(fixed_grid)

    # 5. Quality filter
    # 288 rows * 4 columns = 1152 values. If more than 400 values are missing (~35%), we get rid of the entire day
    missing_count = df_resampled.isna().sum().sum()
    
    if missing_count > 400: # I keep 400 for the moment, the results are OK with that
        # print(f"Missing count:{missing_count} for date: {dates_str}")
        return None

    # 6. INTERPOLATION (Inside and at the edges)
    for col in expected_metrics:
        df_resampled[col] = (
            df_resampled[col]
            .interpolate(method="linear")
            .ffill()  # In case the last value is before 23:55
            .bfill()  # In case the first value is after 00:00
        )

    return df_resampled


def build_dataset(start_date, end_date, db_path):

    result = []
    current = start_date
    all_dates = []

    connection = sqlite3.connect(db_path)

    while current < end_date:
        r = build_daily_matrix(current.isoformat(), connection)
        if r is not None:
            all_dates.append(current.isoformat())
            result.append(r.values)
            
        current = current + timedelta(days=1)

    connection.close()

    final = np.stack(result, dtype=np.float32) # Conversion to float32, better for PyTorch

    return final, all_dates

if __name__ == "__main__":
    start_date = date(2026, 3, 1)
    end_date = date.today()
    final, all_dates = build_dataset(start_date, end_date, "data/garmin_data.db")

    print(final.shape)
    print(final.dtype)