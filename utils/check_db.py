import sqlite3

connection = sqlite3.connect("data/garmin_data.db")
cursor = connection.cursor()

cursor.execute("SELECT * FROM daily_summary")
columns = [description[0] for description in cursor.description]
row = cursor.fetchone()

print("=== daily_summary ===")
for col, val in zip(columns, row):
    print(f"{col}: {val}")

cursor.execute("SELECT metric, COUNT(*) FROM timeseries_data GROUP BY metric")
print("\n=== timeseries_data by metric ===")
for metric, count in cursor.fetchall():
    print(f"{metric}: {count} points")

cursor.execute("SELECT COUNT(*) FROM daily_summary WHERE extracted = 1")
print("\nExtracted days:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM daily_summary WHERE sleep_time_seconds IS NOT NULL")
print("Days with real sleep data:", cursor.fetchone()[0])

cursor.execute("SELECT MIN(date), MAX(date) FROM timeseries_data")
print("Timeseries date range:", cursor.fetchone())

connection.close()