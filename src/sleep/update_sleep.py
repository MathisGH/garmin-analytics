"""
update_sleep.py -- refresh the sleep section of the web app with the latest data.

Goal: after new days have been extracted (backfill.py), rebuild the sleep JSON and
publish it so the website shows the new nights.

How: runs sleep_dataset then precompute_sleep, uploads sleep_precomputed.json to S3,
then updates a Lambda environment variable to force a cold start (the API only reads
the file at import time).
The cross-validated comparison table is not recomputed here: run
`python -m src.sleep.model_comparison` beforehand if you want it refreshed.

Full refresh workflow: backfill.py -> update_scores.py -> update_sleep.py

Usage: python -m src.sleep.update_sleep [--dry-run]
       --dry-run rebuilds the JSON locally but uploads nothing.
"""

import argparse
import subprocess
import sys
import time

import boto3

OUTPUT_PATH = "data/sleep_precomputed.json"
S3_BUCKET = "garmin-anomaly-mathisdurand-2026"
S3_KEY = "sleep_precomputed.json"
LAMBDA_FUNCTION_NAME = "garmin-anomaly-api"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh the sleep section of the web app.")
    parser.add_argument("--dry-run", action="store_true", help="rebuild locally, upload nothing")
    args = parser.parse_args()

    # check=True stops the script if a step fails, so nothing is uploaded from a broken build
    for module in ["src.sleep.sleep_dataset", "src.sleep.precompute_sleep"]:
        print(f"--- Running {module}")
        subprocess.run([sys.executable, "-m", module], check=True)

    if args.dry_run:
        print("Dry run: nothing uploaded.")
        sys.exit()

    boto3.client("s3").upload_file(OUTPUT_PATH, S3_BUCKET, S3_KEY)
    print(f"Uploaded to s3://{S3_BUCKET}/{S3_KEY}")

    lambda_client = boto3.client("lambda")
    # Lambda accepts one configuration change at a time: wait if update_scores.py just made one
    lambda_client.get_waiter("function_updated").wait(FunctionName=LAMBDA_FUNCTION_NAME)
    lambda_client.update_function_configuration(
        FunctionName=LAMBDA_FUNCTION_NAME,
        Environment={"Variables": {"LAST_REFRESH": str(int(time.time()))}},
    )
    print("Lambda cold start triggered")