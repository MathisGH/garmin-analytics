from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)
from dotenv import load_dotenv
import os

TOKEN_STORE = os.path.expanduser("~/.garminconnect")


def get_authenticated_client():
    """Return a logged-in Garmin client, reusing cached tokens when possible"""
    load_dotenv()
    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")

    try:
        print(f"Trying to reuse cached tokens from '{TOKEN_STORE}'...")
        client = Garmin()
        client.login(TOKEN_STORE)
        print("Reused cached session successfully")
    except GarminConnectTooManyRequestsError as err:
        print(f"Rate limited by Garmin, try again later: {err}")
        raise SystemExit(1)
    except (GarminConnectAuthenticationError, GarminConnectConnectionError, FileNotFoundError):
        print("No valid cached session, logging in with credentials...")
        client = Garmin(email=email, password=password)
        client.login(TOKEN_STORE)
        print(f"New tokens saved to '{TOKEN_STORE}'")
    
    return client