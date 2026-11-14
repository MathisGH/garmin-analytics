"""
test_api.py -- tests for src/api.py

Goal: check the HTTP contract the frontend relies on, without any network or real data.

How: api.py reads S3 at import time, so a fixture replaces boto3.client with
a fake S3 serving small synthetic JSON files (mocking with monkeypatch), then imports the module.
FastAPI's TestClient calls the app in memory (no server, no port).
"""

import importlib
import io
import json
import sys

import pytest
from fastapi.testclient import TestClient

FAKE_ANOMALY = {"2026-01-01": {"score_if": -0.5, "score_ae_global": 0.4, "extra": 1}}
FAKE_SLEEP = {
    "meta": {}, "coefficients": {}, "model_comparison": {},
    "nights": [{
        "date": "2026-01-01", "sleep_score": 80, "predicted_score": 78,
        "sleep_time_seconds": 27000, "stages_valid": True, "detail": "x",
    }],
}


class FakeS3:
    """Mimics boto3's S3 client: get_object returns {'Body': <readable>}"""

    # The real call in api.py is: boto3.client("s3").get_object(Bucket=..., Key=...)
    # followed by response["Body"].read(). This fake offers exactly that interface
    # and nothing more: a test double only needs to imitate what the code under test uses.
    def get_object(self, Bucket, Key):
        # Parameter names must match boto3's keyword arguments (Bucket, Key),
        # because api.py calls get_object(Bucket=..., Key=...)
        # Pick the fake JSON according to the requested file
        payload = FAKE_ANOMALY if Key == "scores_precomputed.json" else FAKE_SLEEP
        # json.dumps: dict -> JSON text; .encode(): text -> bytes (S3 returns bytes);
        # io.BytesIO: an in-memory "file" exposing .read(), like the real response Body
        return {"Body": io.BytesIO(json.dumps(payload).encode())}


@pytest.fixture
def client(monkeypatch):
    # 1. api.py only reads S3 when GARMIN_LOCAL != "1". If this variable is set on
    #    my machine, the test would read real local files. Removing it forces the
    #    S3 branch. raising=False: no error if the variable does not exist.
    monkeypatch.delenv("GARMIN_LOCAL", raising=False)

    # 2. Replace boto3.client by a function returning my fake. The lambda accepts
    #    any arguments (*args, **kwargs) because the real call passes "s3".
    #    monkeypatch restores the original boto3.client after the test.
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: FakeS3())

    # 3. Python caches imported modules in sys.modules. api.py runs its S3 reads at
    #    import time, so I remove it from the cache to make Python re-execute it
    #    (with the fake) for each test. Otherwise, a stale copy would be reused.
    monkeypatch.delitem(sys.modules, "src.api", raising=False)

    # 4. Import now (not at the top of the file): the patch must be in place
    #    BEFORE the import triggers load_json(). importlib.import_module imports
    #    from a string, usable anywhere in the code.
    api = importlib.import_module("src.api")

    # 5. TestClient calls the FastAPI app in memory: no server, no port, no network.
    return TestClient(api.app)


def test_day_found(client):
    response = client.get("/days/2026-01-01")
    assert response.status_code == 200
    assert response.json() == FAKE_ANOMALY["2026-01-01"]


def test_day_not_found(client):
    assert client.get("/days/2000-01-01").status_code == 404


def test_days_list_exposes_only_scores(client):
    body = client.get("/days").json()
    assert body == {"2026-01-01": {"score_if": -0.5, "score_ae_global": 0.4}}


def test_sleep_endpoints(client):
    assert set(client.get("/sleep/summary").json()) == {"meta", "coefficients", "model_comparison"}

    nights = client.get("/sleep/nights").json()
    assert set(nights[0]) == {"date", "sleep_score", "predicted_score", "sleep_time_seconds", "stages_valid"}

    assert client.get("/sleep/nights/2026-01-01").status_code == 200
    assert client.get("/sleep/nights/2000-01-01").status_code == 404