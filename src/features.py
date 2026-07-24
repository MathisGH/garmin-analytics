
from datetime import timedelta, date
import time
import pandas as pd
import sqlite3


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


result = []

start = date(2026, 3, 20)
end = date.today()
current = start

connection = sqlite3.connect("data/garmin_data.db")

while current < end:
    r = build_daily_matrix(current.isoformat(), connection)
    if r is not None:
        result.append([current, r])
        
    current = current + timedelta(days=1)

connection.close()

lengths = [len(r) for _, r in result]
print(min(lengths), max(lengths), len(set(lengths)))
print(len(result))