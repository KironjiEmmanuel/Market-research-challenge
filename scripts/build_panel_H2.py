# scripts/build_panel_H2.py
# Builds data/clean/panel_H2_weekly.csv for test card H2.
# Run from the repo root:  python scripts/build_panel_H2.py
#
# Card H2: flow ratio(t) on change in 91-day T-bill yield(t-1), control NSE 20 return(t).
# Week t ends on `period_end` (last trading day of the week).
# The driver for week t is: yield(Monday of week t-1) minus yield(Monday of week t-2).
# Deliberately NO driver-vs-outcome correlations are printed here: this script only
# builds and checks the table. Looking at relationships comes in Phase 3.

import os
import pandas as pd

# ------------------------- EDIT THESE -------------------------
BANK_WEEKLY = "C:/Users/KIRONJI/Desktop/Rock research project/data/clean/bank_flows_weekly.csv"  # output of build_bank_index.py
TBILLS      = "C:/Users/KIRONJI/Desktop/Rock research project/data/raw/treasury_bills.csv"
NSE20       = "C:/Users/KIRONJI/Desktop/Rock research project/data/raw/nse_20_index.csv"
TBILLS_SKIP = 0      # set to 1 or 2 if a "Licensed to..." line sits above the header
NSE20_SKIP  = 0
DAYFIRST    = True   # False if your dates look like 2025-06-24 or 06/24/2025
OUT = "C:/Users/KIRONJI/Desktop/Rock research project/data/clean/panel_H2_weekly.csv"
LOG = "C:/Users/KIRONJI/Desktop/Rock research project/outputs/panel_H2_build_log.txt"
# --------------------------------------------------------------

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def read_rock(path, skiprows=0):
    df = pd.read_csv(path, skiprows=skiprows, thousands=",")
    n0 = len(df)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=DAYFIRST, errors="coerce")
    df = df.dropna(subset=["Date"])
    log(f"{os.path.basename(path)}: {n0} rows read, {n0 - len(df)} dropped (bad dates / stamp lines)")
    return df


# ---------- 1. Weekly bank outcome ----------
bank = pd.read_csv(BANK_WEEKLY)
bank["period_end"] = pd.to_datetime(bank["period_end"])
bank = bank.sort_values("period_end").reset_index(drop=True)
assert bank["period_end"].is_unique, "Duplicate period_end in the weekly bank file"

ratio_cols = [c for c in bank.columns if c.startswith("flow_ratio_")]
panel = pd.DataFrame({
    "period_end": bank["period_end"],
    "flow_ratio": bank["sector_flow_ratio"],              # ratio of sums (sector net flow / sector turnover)
    "flow_ratio_eqw": bank[ratio_cols].mean(axis=1),      # equal-weighted mean of the 5 bank ratios (sensitivity only)
    "n_trading_days": bank["n_trading_days"],
})
log(f"Bank weekly: {len(panel)} weeks, {panel['period_end'].min().date()} -> {panel['period_end'].max().date()}")
log(f"Bank ratio columns averaged for flow_ratio_eqw: {ratio_cols}")

# ---------- 2. 91-day T-bill yield (driver) ----------
tb = read_rock(TBILLS, TBILLS_SKIP)
log("Tenor labels found: " + str(sorted(tb["Tenor"].astype(str).unique())))
log("Source labels found: " + str(sorted(tb["Source"].astype(str).unique())))
t91 = tb[tb["Tenor"].astype(str).str.contains("91")].copy()
labels = t91["Tenor"].astype(str).unique()
assert len(labels) == 1, f"Expected exactly one 91-day label, got {labels}"
t91["Rate (%)"] = pd.to_numeric(t91["Rate (%)"], errors="coerce")
log(f"91-day rows: {len(t91)}, missing rates: {t91['Rate (%)'].isna().sum()}")

# Snap each stamp to the Monday of its week (all should already be Mondays)
t91["week_monday"] = t91["Date"] - pd.to_timedelta(t91["Date"].dt.dayofweek, unit="D")
log(f"91-day dates not stamped on a Monday: {(t91['Date'].dt.dayofweek != 0).sum()}")
assert t91["week_monday"].is_unique, "More than one 91-day rate in the same week: check the Source column"
y = t91.set_index("week_monday")["Rate (%)"].sort_index()

panel["week_monday"] = panel["period_end"] - pd.to_timedelta(panel["period_end"].dt.dayofweek, unit="D")
panel["y91_t1"] = (panel["week_monday"] - pd.Timedelta(days=7)).map(y)    # Monday of week t-1
panel["y91_t2"] = (panel["week_monday"] - pd.Timedelta(days=14)).map(y)   # Monday of week t-2
panel["dy91_lag1"] = panel["y91_t1"] - panel["y91_t2"]                     # percentage points

# ---------- 3. NSE 20 weekly return (control, same week) ----------
n = read_rock(NSE20, NSE20_SKIP)
log("Security labels found: " + str(sorted(n["Security"].astype(str).unique())[:10]))
n = n[n["Security"].astype(str).str.contains(r"nse\s*20|nse20", case=False, regex=True)].copy()
assert n["Security"].nunique() == 1, f"Expected one NSE 20 label, got {n['Security'].unique()}"
n["Price (KES)"] = pd.to_numeric(n["Price (KES)"], errors="coerce")
n = n.dropna(subset=["Price (KES)"]).sort_values("Date")
n_weekend = (n["Date"].dt.dayofweek >= 5).sum()
n = n[n["Date"].dt.dayofweek < 5].drop_duplicates("Date", keep="last")
log(f"NSE 20: {len(n)} weekday rows kept, {n_weekend} weekend rows dropped")

px = pd.merge_asof(
    panel[["period_end"]].sort_values("period_end"),
    n[["Date", "Price (KES)"]],
    left_on="period_end", right_on="Date", direction="backward",
    tolerance=pd.Timedelta(days=5),
)
panel["nse20_px_end"] = px["Price (KES)"].values
panel["nse20_px_prev"] = panel["nse20_px_end"].shift(1)
panel["nse20_ret_pct"] = 100 * (panel["nse20_px_end"] / panel["nse20_px_prev"] - 1)

# ---------- 4. Checks, then drop unusable rows ----------
log("")
log("Rows lost before dropping (reason: count)")
log(f"  flow_ratio missing:   {panel['flow_ratio'].isna().sum()}")
log(f"  dy91_lag1 missing:    {panel['dy91_lag1'].isna().sum()}")
log(f"  nse20_ret_pct missing:{panel['nse20_ret_pct'].isna().sum()}")

final = panel.dropna(subset=["flow_ratio", "dy91_lag1", "nse20_ret_pct"]).reset_index(drop=True)
assert final["flow_ratio"].between(-1, 1).all(), "flow_ratio outside [-1, 1]: check the bank file"

gaps = final["period_end"].diff().dt.days
log("")
log(f"FINAL PANEL: {len(final)} weeks, {final['period_end'].min().date()} -> {final['period_end'].max().date()}")
log(f"Gaps between consecutive weeks > 10 days: {(gaps > 10).sum()}")
if (gaps > 10).any():
    log(final.loc[gaps > 10, "period_end"].dt.date.astype(str).tolist())
log("Trading days per week:")
log(final["n_trading_days"].value_counts().sort_index().to_string())
log("Weeks per year:")
log(final["period_end"].dt.year.value_counts().sort_index().to_string())

# ---------- 5. Spot-check sample (verify by hand against the raw files) ----------
log("")
log("SPOT CHECK: 5 random rows. For each, look up in the raw files:")
log("  y91_t1 and y91_t2 (91-day rate on those two Mondays), and nse20 prices on/before period_end and the previous period_end")
cols = ["period_end", "week_monday", "y91_t1", "y91_t2", "dy91_lag1",
        "nse20_px_prev", "nse20_px_end", "nse20_ret_pct", "flow_ratio"]
log(final.sample(5, random_state=42)[cols].to_string(index=False))

# ---------- 6. Save ----------
os.makedirs(os.path.dirname(OUT), exist_ok=True)
os.makedirs(os.path.dirname(LOG), exist_ok=True)
final.to_csv(OUT, index=False)
with open(LOG, "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved {OUT} and {LOG}")
