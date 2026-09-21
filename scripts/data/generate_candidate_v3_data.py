import os
import pandas as pd
import numpy as np

# Load original datasets
history = pd.read_csv("data/processed/asset_history_simulated.csv", parse_dates=["snapshot_date"])
failures = pd.read_csv("data/processed/failures_simulated.csv", parse_dates=["failure_date"])
original_ml = pd.read_csv("data/processed/failure_ml_dataset.csv", parse_dates=["snapshot_date"])

out_dir = "data/processed/candidate_v3"
os.makedirs(out_dir, exist_ok=True)

# Build corrected target:
# In the original 06_time_split.ipynb:
# history = history[history["snapshot_date"] + pd.Timedelta(days=30) <= latest_failure_date].copy()
# The snapshots are 2019-01-01 to 2026-07-01 (91 snapshots per asset = 91,000 records).
# Let's keep the EXACT SAME 91,000 rows as original_ml, preserving every feature and order,
# but fixing the target column failure_within_30_days so that it looks up failures within next month (<= 32 days).

candidate_ml = original_ml.copy()

# Vectorized lookup of failures within next month (days_diff > 0 and <= 32)
merged = pd.merge(
    candidate_ml[["asset_id", "snapshot_date"]],
    failures[["asset_id", "failure_date"]],
    on="asset_id",
    how="inner"
)
days_diff = (merged["failure_date"] - merged["snapshot_date"]).dt.days
valid_future = (days_diff > 0) & (days_diff <= 32)

corrected_pos = merged[valid_future][["asset_id", "snapshot_date"]].drop_duplicates()
corrected_pos["failure_within_30_days"] = 1

# Update target
candidate_ml = candidate_ml.drop(columns=["failure_within_30_days"]).merge(
    corrected_pos,
    on=["asset_id", "snapshot_date"],
    how="left"
)
candidate_ml["failure_within_30_days"] = candidate_ml["failure_within_30_days"].fillna(0).astype(int)

# Split identically to 06_time_split.ipynb:
# train: < 2025-01-01 (72,000 records)
# validation: 2025-01-01 to 2025-12-01 (12,000 records)
# test: >= 2026-01-01 (7,000 records)
train_v3 = candidate_ml[candidate_ml["snapshot_date"] < "2025-01-01"].copy()
val_v3 = candidate_ml[
    (candidate_ml["snapshot_date"] >= "2025-01-01") &
    (candidate_ml["snapshot_date"] < "2026-01-01")
].copy()
test_v3 = candidate_ml[candidate_ml["snapshot_date"] >= "2026-01-01"].copy()

# Save candidate files
candidate_ml.to_csv(f"{out_dir}/failure_ml_dataset.csv", index=False)
train_v3.to_csv(f"{out_dir}/failure_train.csv", index=False)
val_v3.to_csv(f"{out_dir}/failure_validation.csv", index=False)
test_v3.to_csv(f"{out_dir}/failure_test.csv", index=False)

print("Saved candidate_v3 datasets to", out_dir)
print("Candidate ML dataset shape:", candidate_ml.shape)
print("Train shape:", train_v3.shape, "Positives:", train_v3["failure_within_30_days"].sum())
print("Validation shape:", val_v3.shape, "Positives:", val_v3["failure_within_30_days"].sum())
print("Test shape:", test_v3.shape, "Positives:", test_v3["failure_within_30_days"].sum())
