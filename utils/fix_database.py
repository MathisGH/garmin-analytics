import sqlite3

db_path = "data/garmin_data.db"

cutoff_date = "2026-08-14"

connection = sqlite3.connect(db_path)
cursor = connection.cursor()

cursor.execute("DELETE FROM timeseries_data WHERE date >= ?", (cutoff_date,))
cursor.execute("DELETE FROM daily_summary WHERE date >= ?", (cutoff_date,))

connection.commit()

print(f"Données supprimées à partir du {cutoff_date}.")

connection.close()