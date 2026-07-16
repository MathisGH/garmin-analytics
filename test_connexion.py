from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)
from dotenv import load_dotenv
import os
from datetime import date, timedelta
from src.database import create_schema, DB_PATH
import json
from pathlib import Path

load_dotenv()
GARMIN_EMAIL = os.getenv("GARMIN_EMAIL")
GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")

TOKEN_STORE = os.path.expanduser("~/.garminconnect")  # folder where tokens are cached

try:
    # Try to reuse cached tokens -> no request sent to Garmin's login servers
    print(f"Trying to reuse cached tokens from '{TOKEN_STORE}'...")
    client = Garmin()
    client.login(TOKEN_STORE)
    print("Reused cached session successfully")
except GarminConnectTooManyRequestsError as err:
    # Garmin rate limit -> retrying now would only make it worse
    print(f"Rate limited by Garmin, try again later: {err}")
    raise SystemExit(1)
except (GarminConnectAuthenticationError, GarminConnectConnectionError, FileNotFoundError):
    # No valid cached tokens -> perform a real login with email/password
    print("No valid cached session, logging in with credentials...")
    client = Garmin(email=GARMIN_EMAIL, password=GARMIN_PASSWORD)
    client.login(TOKEN_STORE)  # this call also saves the new tokens to TOKEN_STORE
    print(f"New tokens saved to '{TOKEN_STORE}'")

profile = client.get_full_name()
print("Connection successful -> Name retrieved:", profile)

today = date.today().isoformat()  # format expected by the API: "YYYY-MM-DD"
week_start = (date.today() - timedelta(days=7)).isoformat()
#sleep_data = client.get_sleep_data(today)
#test_data = client.get_steps_data(today)

METHODS = [
    # Daily
    ("get_stats", [today]),
    ("get_user_summary", [today]),
    ("get_stats_and_body", [today]),
    ("get_steps_data", [today]),
    ("get_heart_rates", [today]),
    ("get_resting_heart_rate", [today]),
    ("get_sleep_data", [today]),
    ("get_all_day_stress", [today]),
    ("get_lifestyle_logging_data", [today]),

    # Advanced
    ("get_training_readiness", [today]),
    ("get_morning_training_readiness", [today]),
    ("get_training_status", [today]),
    ("get_respiration_data", [today]),
    ("get_spo2_data", [today]),
    ("get_max_metrics", [today]),
    ("get_hrv_data", [today]),
    ("get_stress_data", [today]),
    ("get_lactate_threshold", []),
    ("get_intensity_minutes_data", [today]),
    ("get_running_tolerance", [week_start, today]),

    # Historical
    ("get_daily_steps", [week_start, today]),
    ("get_body_battery", [week_start, today]),
    ("get_body_battery_events", [week_start]),
    ("get_progress_summary_between_dates", [week_start, today]),
    ("get_weekly_steps", [today]),
    ("get_weekly_stress", [today]),
    ("get_weekly_intensity_minutes", [week_start, today]),
]

OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def summarize(data, indent=0):
    """Print the structure of a nested dict/list without dumping every value"""
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}{key}: {type(value).__name__}")
            summarize(value, indent + 1)
    elif isinstance(data, list):
        print(f"{prefix}[list length={len(data)}]")
        if len(data):
            summarize(data[0], indent + 1)

#for method_name, args in METHODS:
#
#    print("\n" + "=" * 80)
#    print(method_name)
#
#    if not hasattr(client, method_name):
#        print("Method not found.")
#        continue
#
#    try:
#        result = getattr(client, method_name)(*args)
#
#        with open(OUTPUT_DIR / f"{method_name}.json", "w") as f:
#            json.dump(result, f, indent=2)
#
 #       print(type(result))
#
#        if isinstance(result, dict):
#            print(f"Number of keys : {len(result)}")
#            print(f"Top-level keys ({len(result)}):")
#            print(list(result.keys()))
 #       
 #       elif isinstance(result, list):
  #          print(f"Number of elements : {len(result)}")

   #     summarize(result)

#    except Exception as e:
#        print(f"ERROR : {e}")

#heart_rate_data = client.get_heart_rates(today)
#print("=== Heart Rate ===")
#summarize(heart_rate_data)

#stress_data = client.get_stress_data(today)
#print("=== Stress ===")
#summarize(stress_data)

#body_battery_data = client.get_body_battery(today, today)
#print("=== Body Battery ===")
#summarize(body_battery_data)

#print(heart_rate_data["heartRateValueDescriptors"])
#print(heart_rate_data["heartRateValues"][0])  # first point, to see the raw pair


# Test dates going back in time: 1 month, 6 months, 1 year, 2 years
#test_offsets_days = [30, 180, 365, 730]

#for offset in test_offsets_days:
 #   test_date = (date.today() - timedelta(days=offset)).isoformat()
  #  sleep_test = client.get_sleep_data(test_date)
   # 
    # Check if we actually got real data or an empty/null response
    #daily_sleep = sleep_test.get("dailySleepDTO")
    #if daily_sleep and daily_sleep.get("sleepTimeSeconds"):
    #    print(f"{test_date} ({offset} days ago): DATA FOUND — sleepTimeSeconds = {daily_sleep['sleepTimeSeconds']}")
    #else:
    #    print(f"{test_date} ({offset} days ago): NO DATA")


#def find_earliest_sleep_date(client, min_days_ago=730, max_days_ago=3650):
#    """Binary search for the oldest day that still has sleep data.
#    Assumes: min_days_ago has data, max_days_ago does not.
#    Returns the number of days ago of the earliest day with data.
#    """
#    while max_days_ago - min_days_ago > 1:
#        mid_days_ago = (min_days_ago + max_days_ago) // 2
#        test_date = (date.today() - timedelta(days=mid_days_ago)).isoformat()
#
#        sleep_test = client.get_sleep_data(test_date)
#        daily_sleep = sleep_test.get("dailySleepDTO")
#        has_data = bool(daily_sleep and daily_sleep.get("sleepTimeSeconds"))
#
#        print(f"Testing {test_date} ({mid_days_ago} days ago): {'DATA' if has_data else 'NO DATA'}")
#
#        if has_data:
#            min_days_ago = mid_days_ago  # data exists here -> search further back
#        else:
#            max_days_ago = mid_days_ago  # no data here -> search closer to today
#
#    return min_days_ago  # last confirmed day with data

#earliest = find_earliest_sleep_date(client)
#print(f"Earliest data found: {earliest} days ago = {(date.today() - timedelta(days=earliest)).isoformat()}") # -> 2023-12-27

#create_schema(DB_PATH)

#sleep_data = client.get_sleep_data(date.today().isoformat())
#insert_sleep_data(DB_PATH, sleep_data)

from src.database import create_schema, DB_PATH
from src.extract import extract_day
from datetime import date

create_schema(DB_PATH)

today = date.today().isoformat()
extract_day(client, DB_PATH, today)