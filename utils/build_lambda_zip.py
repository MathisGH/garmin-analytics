"""
build_lambda_zip.py -- build a new Lambda deployment package with an updated api.py.

Goal: ship a new version of src/api.py to AWS Lambda without reinstalling the
dependencies (fastapi, pydantic, mangum... are already inside deployment.zip).

How: copy every file of deployment.zip into deployment_v2.zip, except api.py,
which is replaced by the current src/api.py. The old zip is not modified, so it
stays available as a rollback.

Usage (from the project root): python utils/build_lambda_zip.py
"""

import zipfile
from pathlib import Path

OLD_ZIP = Path("deployment.zip")
NEW_ZIP = Path("deployment_v2.zip")
API_SOURCE = Path("src/api.py")

with zipfile.ZipFile(OLD_ZIP) as old, zipfile.ZipFile(NEW_ZIP, "w", zipfile.ZIP_DEFLATED) as new:
    # 1. Copy everything from the old zip except the old api.py
    for entry in old.infolist():
        if entry.filename != "api.py":
            new.writestr(entry, old.read(entry.filename))

    # 2. Add the new api.py at the root of the zip (Lambda handler is "api.handler")
    new.write(API_SOURCE, arcname="api.py")

print(f"Wrote {NEW_ZIP} ({NEW_ZIP.stat().st_size / 1e6:.1f} MB), api.py replaced by {API_SOURCE}")