# scripts/merge_liquidity.py
import glob, os
import pandas as pd

RAW = "C:/Users/KIRONJI/Desktop/Rock research project/data/raw/Liquidity data"
OUT = "C:/Users/KIRONJI/Desktop/Rock research project/data/clean/liquidity_merged.csv"
DATE_COL = "Date"      # check the real header name in one file first
DAYFIRST = True        # set False if your dates look like 2025-06-24 or 06/24/2025

files = sorted(glob.glob(f"{RAW}/*.csv"), key=os.path.getmtime)  # oldest download first
frames = []
for f in files:
    df = pd.read_csv(f, thousands=",")
    n_raw = len(df)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=DAYFIRST, errors="coerce")
    df = df.dropna(subset=[DATE_COL])          # drops the "Licensed to..." stamp lines if they sit in rows
    df["source_file"] = os.path.basename(f)
    print(f"{os.path.basename(f)}: {n_raw} rows, {len(df)} valid, "
          f"{df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}"
          + ("  <-- 1000 rows, probably TRUNCATED" if n_raw >= 1000 else ""))
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

# 1) do overlapping rows agree?
num_cols = [c for c in all_df.select_dtypes("number").columns]
dups = all_df[all_df.duplicated(DATE_COL, keep=False)].sort_values(DATE_COL)
if len(dups):
    spread = dups.groupby(DATE_COL)[num_cols].agg(lambda s: s.max() - s.min()).fillna(0)
    bad = spread[(spread.abs() > 1e-9).any(axis=1)]
    print(f"\nOverlapping dates: {dups[DATE_COL].nunique()}, with conflicting values: {len(bad)}")
    if len(bad): print(bad.head(10))

# 2) keep the most recent download for each date
merged = (all_df.drop_duplicates(DATE_COL, keep="last")
                .sort_values(DATE_COL).reset_index(drop=True)
                .drop(columns="source_file"))

# 3) gap check (weekdays only)
expected = pd.bdate_range(merged[DATE_COL].min(), merged[DATE_COL].max())
missing = expected.difference(merged[DATE_COL])
print(f"\nMerged: {len(merged)} rows, {merged[DATE_COL].min().date()} -> {merged[DATE_COL].max().date()}")
print(f"Missing weekdays: {len(missing)} (holidays are expected; look for long runs)")
print(merged.isna().sum())

os.makedirs("data/clean", exist_ok=True)
merged.to_csv(OUT, index=False)