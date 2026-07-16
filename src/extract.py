from src.database import (
    DB_PATH,
    ensure_daily_row,
    update_daily_summary,
    insert_timeseries,
    METRIC_HEART_RATE,
    METRIC_STRESS,
    METRIC_BODY_BATTERY,
    METRIC_RESPIRATION,
    METRIC_SLEEP_MOVEMENT,
    METRIC_SLEEP_STAGE,
)


def extract_day(client, db_path, day_date):
    """Fetch and store all retained metrics for one day (format: 'YYYY-MM-DD')"""
    ensure_daily_row(db_path, day_date)
    total_points = 0

    # Daily summary fields
    stats = client.get_stats_and_body(day_date)
    update_daily_summary(db_path, day_date, {
        "steps": stats.get("totalSteps"),
        "calories": stats.get("totalKilocalories"),
        "distance_meters": stats.get("totalDistanceMeters"),
        "resting_hr": stats.get("restingHeartRate"),
        "min_hr": stats.get("minHeartRate"),
        "max_hr": stats.get("maxHeartRate"),
        "avg_stress": stats.get("averageStressLevel"),
        "max_stress": stats.get("maxStressLevel"),
        "body_battery_charged": stats.get("bodyBatteryChargedValue"),
        "body_battery_drained": stats.get("bodyBatteryDrainedValue"),
        "body_battery_highest": stats.get("bodyBatteryHighestValue"),
        "body_battery_lowest": stats.get("bodyBatteryLowestValue"),
        "body_battery_at_wake": stats.get("bodyBatteryAtWakeTime"),
        "avg_spo2": stats.get("averageSpo2"),
        "lowest_spo2": stats.get("lowestSpo2"),
        "avg_respiration": stats.get("avgWakingRespirationValue"),
    })

    sleep_data = client.get_sleep_data(day_date)
    daily_sleep = sleep_data.get("dailySleepDTO")
    if daily_sleep and daily_sleep.get("sleepTimeSeconds"):
        update_daily_summary(db_path, day_date, {
            "sleep_score": daily_sleep.get("sleepScores", {}).get("overall", {}).get("value"),
            "sleep_time_seconds": daily_sleep.get("sleepTimeSeconds"),
            "deep_sleep_seconds": daily_sleep.get("deepSleepSeconds"),
            "light_sleep_seconds": daily_sleep.get("lightSleepSeconds"),
            "rem_sleep_seconds": daily_sleep.get("remSleepSeconds"),
            "awake_sleep_seconds": daily_sleep.get("awakeSleepSeconds"),
            "body_battery_change": sleep_data.get("bodyBatteryChange"),
        })
        total_points += insert_timeseries(
            db_path, day_date, METRIC_SLEEP_MOVEMENT,
            sleep_data.get("sleepMovement", []), "startGMT", "activityLevel",
        )
        total_points += insert_timeseries(
            db_path, day_date, METRIC_SLEEP_STAGE,
            sleep_data.get("sleepLevels", []), "startGMT", "activityLevel",
        )

    # Training status: nested under a dynamic device ID, so we grab the first device found
    training = client.get_training_status(day_date)
    vo2max = training.get("mostRecentVO2Max", {}).get("generic", {}).get("vo2MaxValue")
    load_map = training.get("mostRecentTrainingLoadBalance", {}).get("metricsTrainingLoadBalanceDTOMap", {})
    status_map = training.get("mostRecentTrainingStatus", {}).get("latestTrainingStatusData", {})
    first_load = next(iter(load_map.values()), {})
    first_status = next(iter(status_map.values()), {})
    update_daily_summary(db_path, day_date, {
        "vo2max": vo2max,
        "training_load_weekly": first_load.get("monthlyLoadAerobicLow"),
        "training_status": first_status.get("trainingStatus"),
    })

    intensity = client.get_intensity_minutes_data(day_date)
    update_daily_summary(db_path, day_date, {
        "intensity_minutes_moderate": intensity.get("moderateMinutes"),
        "intensity_minutes_vigorous": intensity.get("vigorousMinutes"),
    })

    # Timeseries
    hr_data = client.get_heart_rates(day_date)
    total_points += insert_timeseries(
        db_path, day_date, METRIC_HEART_RATE,
        [{"t": p[0], "v": p[1]} for p in hr_data.get("heartRateValues") or []],
        "t", "v",
    )

    stress_data = client.get_stress_data(day_date)
    total_points += insert_timeseries(
        db_path, day_date, METRIC_STRESS,
        [{"t": p[0], "v": p[1]} for p in stress_data.get("stressValuesArray") or [] if p[1] is not None and p[1] >= 0],
        "t", "v",
    )
    total_points += insert_timeseries(
        db_path, day_date, METRIC_BODY_BATTERY,
        [{"t": p[0], "v": p[1]} for p in stress_data.get("bodyBatteryValuesArray") or [] if p[1] is not None],
        "t", "v",
    )

    respiration_data = client.get_respiration_data(day_date)
    total_points += insert_timeseries(
        db_path, day_date, METRIC_RESPIRATION,
        [{"t": p[0], "v": p[1]} for p in respiration_data.get("respirationValuesArray") or [] if p[1] is not None and p[1] >= 0],
        "t", "v",
    )

    print(f"Extracted {day_date}: daily summary updated, {total_points} timeseries points")