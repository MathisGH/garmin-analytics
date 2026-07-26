from features import build_dataset
from datetime import date



start_date = date(2026, 3, 1)
end_date = date.today()
final, all_dates = build_dataset(start_date, end_date, "data/garmin_data.db")

# train/test split
cutoff = int(0.85*len(all_dates))

final_train = final[:cutoff]
dates_train = all_dates[:cutoff]

final_val = final[cutoff:]
dates_val = all_dates[cutoff:]

print(final_train.shape)
print(len(dates_train))

for l in [final_train, dates_train, final_val, dates_val]:
    print(f"First value: {l[0]}")
    print(f"Last value: {l[-1]}")


