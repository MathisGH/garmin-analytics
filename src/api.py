"""
api.py -- ... .

Goal: API which will allow the user to compare metrics from each day

How: Using FastAPI, 

Launching using: 'uvicorn src.api:app --reload'
"""

from fastapi import FastAPI, HTTPException
import json

app = FastAPI()

######## à décrire à quoi ça sert très rapidement
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
#########

with open("data/scores_precomputed.json", 'r') as file:
    data = json.load(file)


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