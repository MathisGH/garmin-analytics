import sqlite3

connection = sqlite3.connect("data/garmin_data.db")
cursor = connection.cursor()

cursor.execute("SELECT * FROM daily_summary")
print("=== daily_summary ===")
for row in cursor.fetchall():
    print(row)

cursor.execute("SELECT COUNT(*) FROM timeseries_data")
print("\n=== timeseries_data count ===")
print(cursor.fetchone())

connection.close()