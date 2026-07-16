import sqlite3
import os

DB_PATH = os.path.join("data", "garmin_data.db")

# Constants for metric names, to avoid typos like "Heart_Rate" vs "heart_rate"
METRIC_HEART_RATE = "heart_rate"
METRIC_STRESS = "stress"
METRIC_BODY_BATTERY = "body_battery"
METRIC_RESPIRATION = "respiration"
METRIC_SLEEP_MOVEMENT = "sleep_movement"
METRIC_SLEEP_STAGE = "sleep_stage"



def create_schema(db_path):
    """Create the database tables if they do not already exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True) # Create the directory if it does not already exist

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY,
            steps INTEGER,
            calories REAL,
            distance_meters REAL,
            resting_hr INTEGER,
            min_hr INTEGER,
            max_hr INTEGER,
            avg_stress INTEGER,
            max_stress INTEGER,
            body_battery_charged INTEGER,
            body_battery_drained INTEGER,
            body_battery_highest INTEGER,
            body_battery_lowest INTEGER,
            body_battery_at_wake INTEGER,
            avg_respiration REAL,
            sleep_score INTEGER,
            sleep_time_seconds INTEGER,
            deep_sleep_seconds INTEGER,
            light_sleep_seconds INTEGER,
            rem_sleep_seconds INTEGER,
            awake_sleep_seconds INTEGER,
            body_battery_change INTEGER,
            vo2max REAL,
            training_load_weekly REAL,
            training_status INTEGER,
            intensity_minutes_moderate INTEGER,
            intensity_minutes_vigorous INTEGER,
            extracted INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS timeseries_data (
            date TEXT,
            metric TEXT,
            timestamp INTEGER,
            value REAL,
            FOREIGN KEY (date) REFERENCES daily_summary(date);
            UNIQUE(date, metric, timestamp)
        )
    """)

    connection.commit()
    connection.close()
    print(f"Schema created successfully in '{db_path}'")


def ensure_daily_row(db_path, day_date):
    """Make sure a daily_summary row exists for this date, so timeseries
    inserts never violate the foreign key, even if some endpoint fails
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO daily_summary (date) VALUES (?)", (day_date,)
    )
    connection.commit()
    connection.close()


def update_daily_summary(db_path, day_date, fields):
    """Update only the given columns (a dict of column_name: value) for one date"""
    if not fields:
        return
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    set_clause = ", ".join(f"{column} = ?" for column in fields)
    values = list(fields.values()) + [day_date]

    cursor.execute(f"UPDATE daily_summary SET {set_clause} WHERE date = ?", values)
    connection.commit()
    connection.close()


def insert_timeseries(db_path, day_date, metric, points, time_key, value_key):
    """Insert a list of raw Garmin points (list of dicts) as timeseries rows"""
    rows = [(day_date, metric, point[time_key], point[value_key]) for point in points]
    if not rows:
        return 0

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.executemany(
        "INSERT OR IGNORE INTO timeseries_data (date, metric, timestamp, value) VALUES (?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()
    return len(rows)