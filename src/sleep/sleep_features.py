"""
sleep_features.py -- build the day-D-causes -> night-D+1-sleep dataset

Goal: join daily_summary activity/stress features (day D) with the sleep
outcome of the following night (day D+1), adding EMA-smoothed versions of
a few features to capture accumulated fatigue, not just the previous day
"""

from datetime import timedelta, date
import pandas as pd
import sqlite3
import numpy as np

cause_metrics = ["steps", "calories", "distance_meters", "avg_stress", "max_stress",
    "resting_hr", "min_hr", "max_hr", "body_battery_charged",
    "body_battery_drained", "body_battery_highest", "body_battery_lowest",
    "avg_respiration",
    "intensity_minutes_moderate", "intensity_minutes_vigorous", "vo2max",
    "avg_stress_ema3", "avg_stress_ema7", "avg_stress_ema14", "training_load_ema3",
    "training_load_ema7", "training_load_ema14", "training_load_ema28",
    "intensity_minutes_vigorous_ema3", "intensity_minutes_vigorous_ema7",
    "intensity_minutes_vigorous_ema14", "calories_ema3", "calories_ema7",
    "calories_ema14", "training_load_acwr"]
effect_metrics = ["sleep_score", "sleep_time_seconds", "deep_sleep_seconds",
                  "light_sleep_seconds", "rem_sleep_seconds", "awake_sleep_seconds"]

# Maps each EMA cause_metrics name to (source_column, span)
EMA_SPECS = {
    "avg_stress_ema3": ("avg_stress", 3),
    "avg_stress_ema7": ("avg_stress", 7),
    "avg_stress_ema14": ("avg_stress", 14),
    "training_load_ema3": ("training_load_weekly", 3),
    "training_load_ema7": ("training_load_weekly", 7),
    "training_load_ema14": ("training_load_weekly", 14),
    "training_load_ema28": ("training_load_weekly", 28),
    "intensity_minutes_vigorous_ema3": ("intensity_minutes_vigorous", 3),
    "intensity_minutes_vigorous_ema7": ("intensity_minutes_vigorous", 7),
    "intensity_minutes_vigorous_ema14": ("intensity_minutes_vigorous", 14),
    "calories_ema3": ("calories", 3),
    "calories_ema7": ("calories", 7),
    "calories_ema14": ("calories", 14),
}


def build_sleep_dataset(db_path: str) -> pd.DataFrame:
    connection = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM daily_summary ORDER BY date", con=connection)
    connection.close()

    df["date"] = pd.to_datetime(df["date"])

    # EMA features, computed BEFORE the shift
    for ema_col, (source_col, span) in EMA_SPECS.items():
        df[ema_col] = df[source_col].ewm(span=span, adjust=False).mean()

    # Acute:Chronic Workload Ratio (ACWR) -> relative training load surge
    df["training_load_acwr"] = df["training_load_ema7"] / df["training_load_ema28"]

    base_cause_metrics = [m for m in cause_metrics if m not in EMA_SPECS]
    all_cause_metrics = base_cause_metrics + list(EMA_SPECS.keys())

    # Shift causes by 1 day so day D's values line up with the night D->D+1
    causes = df[all_cause_metrics].shift(1)
    causes.columns = [f"{col}" for col in all_cause_metrics]

    effects = df[effect_metrics]

    result = pd.concat([df["date"], causes, effects], axis=1)
    result = result.dropna().reset_index(drop=True)

    return result


if __name__ == "__main__":
    dataset = build_sleep_dataset("data/garmin_data.db")
    print(dataset.shape)
    print(dataset.describe())
    print(dataset.head())