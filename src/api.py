"""
api.py -- API serving the anomaly scores and the sleep analysis of the Garmin web app.

Goal: let the frontend explore anomaly scores per day and sleep nights (scores,
explanations, model comparison) without computing anything at request time.

How: FastAPI app. The data is precomputed offline (scores_precomputed.json and
sleep_precomputed.json stored in AWS S3) and read ONCE at import time: from S3
when running on AWS Lambda, from the local data/ folder when GARMIN_LOCAL=1
(local development). On Lambda, the files are therefore only re-read on a cold start.

Launching locally (PowerShell): $env:GARMIN_LOCAL="1"; uvicorn src.api:app --reload
"""

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI()

######## adding the CORS headers for the Lambda response to the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
#########

S3_BUCKET = "garmin-anomaly-mathisdurand-2026"
ANOMALY_KEY = "scores_precomputed.json"
SLEEP_KEY = "sleep_precomputed.json"
USE_LOCAL = os.environ.get("GARMIN_LOCAL") == "1"


def load_json(key: str) -> dict:
    """Read a precomputed JSON file: local data/ folder in dev, S3 bucket on Lambda."""
    if USE_LOCAL:
        with open(f"data/{key}", "r", encoding="utf-8") as file:
            return json.load(file)
    import boto3  # imported here so local development does not need AWS credentials

    response = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(response["Body"].read())


data = load_json(ANOMALY_KEY)
sleep_data = load_json(SLEEP_KEY)
sleep_nights = {night["date"]: night for night in sleep_data["nights"]}  # lookup by date

NIGHT_LIST_FIELDS = ("date", "sleep_score", "predicted_score", "sleep_time_seconds", "stages_valid")


########## Anomaly detection ##########

@app.get("/days/{day_date}")  # Date format : YYYY-MM-DD
def get_data(day_date: str) -> dict:
    if day_date not in data:
        raise HTTPException(status_code=404, detail=f"No data for {day_date}")
    return data[day_date]


@app.get("/days")
def list_days() -> dict:
    return {
        day: {
            "score_if": entry["score_if"],
            "score_ae_global": entry["score_ae_global"],
        }
        for day, entry in data.items()
    }


########## Sleep ##########

@app.get("/sleep/summary")
def sleep_summary() -> dict:
    """Model description, global coefficients and cross-validated model comparison"""
    return {key: sleep_data[key] for key in ("meta", "coefficients", "model_comparison")}


@app.get("/sleep/nights")
def list_nights() -> list:
    """Light list of all nights (for the overview chart / date picker)"""
    return [{field: night[field] for field in NIGHT_LIST_FIELDS} for night in sleep_data["nights"]]


@app.get("/sleep/nights/{night_date}")  # Date format : YYYY-MM-DD
def get_night(night_date: str) -> dict:
    """Full detail of one night: stages, feature values and contribution of each feature"""
    if night_date not in sleep_nights:
        raise HTTPException(status_code=404, detail=f"No sleep data for {night_date}")
    return sleep_nights[night_date]


handler = Mangum(app, lifespan="off") # "lifespan" argument in case that startup/shutdown events are used (not this case here because the data is loaded at the import)