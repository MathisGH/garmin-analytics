"""Quick check: compare dates in the sleep JSON with the database."""
import json
import numpy as np
import pandas as pd

dates = pd.date_range("2026-01-01", periods=20)
values = np.arange(1, 21)
#df = pd.DataFrame({"date": dates, "value": values})

print(len(dates))
print(len(values))