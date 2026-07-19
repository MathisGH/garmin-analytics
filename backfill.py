from src.auth import get_authenticated_client
from src.database import create_schema, DB_PATH, is_day_already_extracted
from src.extract import extract_day
from datetime import timedelta, date
import time

start = date(2023, 10, 1) # Approximately the date when I started to wear my Garmin watch everyday
end = date.today() - timedelta(days=1) # We don't want today's data because the day is not over yet -> incomplete data
current = start
all_dates = []

while current < end:
    all_dates.append(current.isoformat())
    current = current + timedelta(days=1)


client = get_authenticated_client()
create_schema(DB_PATH)

success_count = 0
skip_count = 0
error_count = 0

for day_date in all_dates:
    if is_day_already_extracted(DB_PATH, day_date):
        skip_count += 1
        continue

    try:
        extract_day(client, DB_PATH, day_date)
        success_count += 1
    except Exception as e:
        print(f"ERROR on {day_date}: {e}")
        error_count += 1

    time.sleep(0.5)

print(f"Done. Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")