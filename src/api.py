"""
api.py -- ... .

Goal: API which will allow the user to compare metrics from each day

How: Using FastAPI, 

Launching using: 'uvicorn src.api:app --reload'
"""

from fastapi import FastAPI, HTTPException
import json

app = FastAPI()

with open("data/scores_precomputed.json", 'r') as file:
    data = json.load(file)


@app.get("/days/{day_date}") # Date format : YYYY-MM-DD
def get_data(day_date: str) -> dict:
    if day_date not in data:
        raise HTTPException(status_code=404, detail=f"No data for {day_date}")
    return data[day_date]