"""
api.py -- ... .

Goal: API which will allow the user to compare metrics from each day

How: Using FastAPI, 

Launching using: 'uvicorn src.api:app --reload'
"""

from fastapi import FastAPI, HTTPException
import json

app = FastAPI()

######## adding the CORS headers for the Lambda response to the browser
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
#########

## Local version *start* ##
#with open("data/scores_precomputed.json", 'r') as file:
#    data = json.load(file)
## Local version *end* ##

## AWS with S3 version *start* ##
import boto3

S3_BUCKET = "garmin-anomaly-mathisdurand-2026"
S3_KEY = "scores_precomputed.json"

s3_client = boto3.client("s3")
response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
data = json.loads(response["Body"].read())
## AWS with S3 version *end* ##

@app.get("/days/{day_date}") # Date format : YYYY-MM-DD
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

from mangum import Mangum

handler = Mangum(app, lifespan="off", api_gateway_base_path="/") # lifespan=off et api_gateway_base_path="/" : pour éviter certains problèmes ?