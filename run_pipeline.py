"""
run_pipeline.py -- daily update of the Garmin project.

Goal: chain the three update steps (extract new Garmin days, refresh the
anomaly scores, refresh the sleep analysis) so a scheduler can run them
unattended.

How: each step runs as its own process (the same commands I used by hand),
in order, from the project root. Output goes to logs/pipeline.log. The
pipeline stops at the first failing step and exits with code 1, so the
scheduler and the log can tell that something went wrong.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "logs" / "pipeline.log"

# sys.executable is the Python running this script (the venv one), so every
# step uses the same environment
STEPS = [
    ("extract", [sys.executable, "backfill.py"]),
    ("anomaly scores", [sys.executable, "-m", "src.anomaly.update_scores"]),
    ("sleep analysis", [sys.executable, "-m", "src.sleep.update_sleep"]),
]


def run_step(name, command, log_file):
    """Run one step, write its output to the log, return True if it succeeded"""
    log_file.write(f"\n=== {name} -- started {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
    log_file.flush()  # write our lines before the child process writes its own

    start = time.time()
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,  # the scripts use relative paths (data/, mlflow.db)
        stdout=log_file,
        stderr=subprocess.STDOUT,  # errors go to the same log, in order
        env={**os.environ, "PYTHONUTF8": "1"},  # avoid encoding crashes when output is redirected
    )
    duration = time.time() - start

    status = "OK" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    log_file.write(f"=== {name} -- {status} in {duration:.0f}s ===\n")
    log_file.flush()
    return result.returncode == 0


def main():
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n##### Pipeline run {datetime.now():%Y-%m-%d %H:%M:%S} #####\n")
        for name, command in STEPS:
            if not run_step(name, command, log_file): # If run_step returns 0 (False), then return 1
                log_file.write("Pipeline stopped: a step failed.\n")
                return 1
        log_file.write("Pipeline finished: all steps OK.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())













