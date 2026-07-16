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

connection.close()