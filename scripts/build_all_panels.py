# scripts/build_all_panels.py   (v2: matched to the real repo file formats)
# Builds the panels required by the test cards (docs/test_cards.md) into data/clean/:
#   panel_H1_weekly.csv, panel_H2_weekly.csv, panel_H3_monthly.csv,
#   panel_H4_weekly.csv (only if the merged liquidity file exists), panel_H5_monthly.csv
# Run from the repo root:  python scripts/build_all_panels.py
#
# No driver-vs-outcome correlations are printed anywhere: relationships are explored in
# Phase 3, after the cards are final. This script only builds and checks tables.

import os
import sys
import numpy as np
import pandas as pd

# ======================== PATHS AND COLUMNS (match the repo) ========================
BANK_FLOWS_W  = "C:/Users/KIRONJI/Desktop/Rock research project/data/clean/bank_flows_weekly.csv"  # period_end, sector_flow_ratio, ...
BANK_FLOWS_M  = "C:/Users/KIRONJI/Desktop/Rock research project/data/clean/bank_flows_monthly.csv"
BANK_RET_M    = "C:/Users/KIRONJI/Desktop/Rock research project/data/clean/bank_returns_monthly.csv"  # ew_ret = equal-weighted index return (decimal)
TBILLS        = "C:/Users/KIRONJI/Desktop/Rock research project/data/raw/treasury_bills.csv"
NSE20         = "C:/Users/KIRONJI/Desktop/Rock research project/data/raw/nse_20_index.csv"
FX            = "C:/Users/KIRONJI/Desktop/Rock research project/data/raw/exchange_rates.csv"
INFL_HEAD     = "C:/Users/KIRONJI/Desktop/Rock research project/data/raw/head inflation data.csv"
INFL_CORE     = "C:/Users/KIRONJI/Desktop/Rock research project/data/raw/core inflation data.csv"
DEBT          = "C:/Users/KIRONJI/Desktop/Rock research project/data/raw/public_debt.csv"
LIQUIDITY     = "C:/Users/KIRONJI/Desktop/Rock research project/data/clean/liquidity_merged.csv"      # LONG format: Date, Indicator, Value (H4 only)
LIQ_RATE_NAME = "Daily Average Interbank Rate"
LIQ_VOL_NAME  = "Daily Volumes (KES Mn)"               # descriptive only
OUT_DIR       = "C:/Users/KIRONJI/Desktop/Rock research project/data/clean"
LOG           = "C:/Users/KIRONJI/Desktop/Rock research project/outputs/panels_build_log.txt"
H1_SPLIT      = pd.Timestamp("2024-01-22")             # fixed on card H1
H5_END        = pd.Timestamp("2025-04-30")             # fixed on card H5
# =====================================================================================

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


# ------------------------------------------------------------------ helpers
def num_text(s):
    """Raw Rock text -> float: strips %, 'KES ', commas, plus signs. '-' alone becomes NaN."""
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def pdate(s):
    """Rock dates come as 'Jan 12, 2015', 'June 25, 2026', 'Apr 2025', '6/29/2026' (month first)."""
    return pd.to_datetime(s, format="mixed", dayfirst=False, errors="coerce")


def load_raw(path):
    df = pd.read_csv(path, comment="#", skip_blank_lines=True, dtype=str)
    n0 = len(df)
    df["Date"] = pdate(df["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    log(f"{os.path.basename(path)}: {n0} rows read, {n0 - len(df)} dropped (unparseable dates)")
    return df


def load_clean(path):
    df = pd.read_csv(path)
    df["period_end"] = pd.to_datetime(df["period_end"])
    return df.sort_values("period_end").reset_index(drop=True)


def last_on_or_before(dates, src, valcol, tol_days=5):
    left = pd.DataFrame({"d": pd.to_datetime(dates.to_numpy())}).reset_index().sort_values("d")
    r = pd.merge_asof(left, src[["Date", valcol]].sort_values("Date"), left_on="d", right_on="Date",
                      direction="backward", tolerance=pd.Timedelta(days=tol_days))
    return r.sort_values("index")[valcol].to_numpy()


def load_nse():
    n = load_raw(NSE20)
    n = n[n["Security"].str.contains(r"nse\s*20", case=False, regex=True)].copy()
    assert n["Security"].nunique() == 1, f"Expected one NSE 20 label, got {n['Security'].unique()}"
    n["px"] = num_text(n["Price (KES)"])
    n = n.dropna(subset=["px"])
    wk = (n["Date"].dt.dayofweek >= 5).sum()
    n = n[n["Date"].dt.dayofweek < 5].drop_duplicates("Date", keep="last")
    log(f"NSE 20: {len(n)} weekday rows ({n['Date'].min().date()} -> {n['Date'].max().date()}), {wk} weekend rows dropped")
    return n[["Date", "px"]].rename(columns={"px": "Price"})


def add_nse_return(panel, nse, tol_days, monthly=False):
    panel["nse20_px_end"] = last_on_or_before(panel["period_end"], nse, "Price", tol_days)
    panel["nse20_px_prev"] = panel["nse20_px_end"].shift(1)
    first = panel["period_end"].iloc[0]  # NSE 20 history starts before our window: fetch the previous price
    prev_date = (first.to_period("M") - 1).end_time.normalize() if monthly else first - pd.Timedelta(days=7)
    panel.loc[panel.index[0], "nse20_px_prev"] = last_on_or_before(pd.Series([prev_date]), nse, "Price", tol_days)[0]
    panel["nse20_ret_pct"] = 100 * (panel["nse20_px_end"] / panel["nse20_px_prev"] - 1)


def add_week_monday(panel):
    panel["week_monday"] = panel["period_end"] - pd.to_timedelta(panel["period_end"].dt.dayofweek, unit="D")


def monthly_series(df, valcol, name):
    m = df["Date"].dt.to_period("M")
    log(f"{name}: {df['Date'].min().date()} -> {df['Date'].max().date()}, "
        f"stamp day-of-month most common: {df['Date'].dt.day.value_counts().head(4).to_dict()}")
    assert not m.duplicated().any(), f"{name}: two rows fall in the same month"
    s = pd.Series(df[valcol].to_numpy(), index=pd.PeriodIndex(m, freq="M")).sort_index()
    full = pd.period_range(s.index.min(), s.index.max(), freq="M")
    s = s.reindex(full)
    since = s.loc["2017-01":]
    log(f"{name}: missing months since 2017-01: {since.isna().sum()}  {since[since.isna()].index.astype(str).tolist()[:8]}")
    return s


def lag_lookup(s, months, k):
    return months.map(lambda p: s.get(p - k, np.nan)).astype(float)


def finish(panel, name, outfile, need, spot_cols, gap_days, end=None, split_date=None):
    log(f"\n==================== {name} ====================")
    for c in need:
        log(f"  rows with missing {c}: {panel[c].isna().sum()}")
    f = panel.dropna(subset=need)
    if end is not None:
        f = f[f["period_end"] <= end]
    f = f.reset_index(drop=True)
    assert f["period_end"].is_unique, f"{name}: duplicate period_end"
    if "flow_ratio" in f.columns:
        assert f["flow_ratio"].between(-1, 1).all(), f"{name}: flow_ratio outside [-1, 1]"
    if split_date is not None:
        f["split_part"] = np.where(f["period_end"] <= split_date, 1, 2)
    else:
        f["split_part"] = np.where(np.arange(len(f)) < len(f) / 2, 1, 2)
    gaps = f["period_end"].diff().dt.days
    log(f"FINAL: {len(f)} rows, {f['period_end'].min().date()} -> {f['period_end'].max().date()}")
    log(f"Gaps between consecutive rows > {gap_days} days: {(gaps > gap_days).sum()}"
        + (f"  at {f.loc[gaps > gap_days, 'period_end'].dt.date.astype(str).tolist()[:10]}" if (gaps > gap_days).any() else ""))
    p2 = f.loc[f["split_part"] == 2, "period_end"].min()
    log(f"Split check: part 1 = {(f['split_part'] == 1).sum()} rows, part 2 = {(f['split_part'] == 2).sum()} rows (part 2 starts {p2.date()})")
    log("Rows per year: " + str(f["period_end"].dt.year.value_counts().sort_index().to_dict()))
    log(f"SPOT CHECK {name}: 5 random rows (verify by hand against the raw files)")
    log(f.sample(min(5, len(f)), random_state=42)[spot_cols].to_string(index=False))
    os.makedirs(OUT_DIR, exist_ok=True)
    f.to_csv(os.path.join(OUT_DIR, outfile), index=False)
    log(f"Saved {os.path.join(OUT_DIR, outfile)}")
    return f


def weekly_base():
    b = load_clean(BANK_FLOWS_W)
    ratio_cols = [c for c in b.columns if c.startswith("flow_ratio_")]
    panel = pd.DataFrame({
        "period_end": b["period_end"],
        "flow_ratio": pd.to_numeric(b["sector_flow_ratio"], errors="coerce"),
        "flow_ratio_eqw": b[ratio_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1),
        "sector_turnover_kes": pd.to_numeric(b["sector_turnover_kes"], errors="coerce"),
        "n_trading_days": pd.to_numeric(b["n_trading_days"], errors="coerce"),
    })
    assert panel["period_end"].is_unique
    add_week_monday(panel)
    log(f"Weekly bank base: {len(panel)} weeks; weekday of period_end: "
        f"{panel['period_end'].dt.dayofweek.value_counts().sort_index().to_dict()} (4 = Friday)")
    return panel


# ------------------------------------------------------------------ H1
def build_H1(nse):
    panel = weekly_base()
    fx = load_raw(FX)
    fx = fx[fx["Currency_pair"].str.upper().str.contains("USD/KES")].copy()
    fx["rate"] = num_text(fx["Exchange Rate"])
    log("FX weekday counts (0=Mon ... 6=Sun): " + str(fx["Date"].dt.dayofweek.value_counts().sort_index().to_dict())
        + "  -> weekend rows are dropped")
    fx = fx.dropna(subset=["rate"])
    fx = fx[fx["Date"].dt.dayofweek < 5].drop_duplicates("Date", keep="last")
    pk = fx.loc[fx["rate"].idxmax()]
    log(f"USD/KES: {len(fx)} weekday rows, {fx['Date'].min().date()} -> {fx['Date'].max().date()}, "
        f"range {fx['rate'].min():.2f} - {fx['rate'].max():.2f}, peak {pk['rate']:.2f} on {pk['Date'].date()}")
    fxs = fx[["Date", "rate"]]
    panel["fx_end"] = last_on_or_before(panel["period_end"], fxs, "rate", 5)
    panel["fx_chg_pct"] = 100 * (panel["fx_end"] / panel["fx_end"].shift(1) - 1)
    panel["dfx_lag1_pct"] = panel["fx_chg_pct"].shift(1)
    add_nse_return(panel, nse, 5)
    return finish(panel, "H1 weekly", "panel_H1_weekly.csv",
                  need=["flow_ratio", "dfx_lag1_pct", "nse20_ret_pct"],
                  spot_cols=["period_end", "fx_end", "fx_chg_pct", "dfx_lag1_pct", "nse20_ret_pct", "flow_ratio"],
                  gap_days=10, split_date=H1_SPLIT)


# ------------------------------------------------------------------ H2
def build_H2(nse):
    panel = weekly_base()
    tb = load_raw(TBILLS)
    log("T-bill tenor labels: " + str(sorted(tb["Tenor"].unique())))
    t = tb[tb["Tenor"].str.contains("91")].copy()
    assert t["Tenor"].nunique() == 1
    t["rate"] = num_text(t["Rate (%)"])
    t["week_monday"] = t["Date"] - pd.to_timedelta(t["Date"].dt.dayofweek, unit="D")
    t["is_mon"] = t["Date"].dt.dayofweek == 0
    n_off = (~t["is_mon"]).sum()
    t = t.sort_values(["week_monday", "is_mon"], ascending=[True, False])
    n_dup = t["week_monday"].duplicated().sum()
    t = t.drop_duplicates("week_monday")
    log(f"91-day rows: {len(t) + n_dup}; stamped off-Monday: {n_off}; two rows in one week (kept the Monday one): {n_dup}; "
        f"missing rates: {t['rate'].isna().sum()}")
    y = t.set_index("week_monday")["rate"].sort_index()
    panel["y91_t1"] = (panel["week_monday"] - pd.Timedelta(days=7)).map(y)
    panel["y91_t2"] = (panel["week_monday"] - pd.Timedelta(days=14)).map(y)
    panel["dy91_lag1"] = panel["y91_t1"] - panel["y91_t2"]
    add_nse_return(panel, nse, 5)
    return finish(panel, "H2 weekly", "panel_H2_weekly.csv",
                  need=["flow_ratio", "dy91_lag1", "nse20_ret_pct"],
                  spot_cols=["period_end", "week_monday", "y91_t1", "y91_t2", "dy91_lag1", "nse20_ret_pct", "flow_ratio"],
                  gap_days=10)


# ------------------------------------------------------------------ H3
def build_H3(nse):
    b = load_clean(BANK_RET_M)
    panel = pd.DataFrame({"period_end": b["period_end"], "bank_ret": pd.to_numeric(b["ew_ret"], errors="coerce")})
    panel["month"] = pd.PeriodIndex(panel["period_end"], freq="M")
    r = panel["bank_ret"]
    log(f"Monthly equal-weighted bank return: min {r.min():.4f}, max {r.max():.4f}, mean {r.mean():.4f} (decimal: 0.05 = 5%)")
    head_df, core_df = load_raw(INFL_HEAD), load_raw(INFL_CORE)
    head_df["v"], core_df["v"] = num_text(head_df["Rate (%)"]), num_text(core_df["Rate (%)"])
    head, core = monthly_series(head_df, "v", "Headline inflation"), monthly_series(core_df, "v", "Core inflation")
    panel["infl_head_t1"] = lag_lookup(head, panel["month"], 1)
    panel["infl_head_t2"] = lag_lookup(head, panel["month"], 2)
    panel["dinfl_head_lag1"] = panel["infl_head_t1"] - panel["infl_head_t2"]
    panel["dinfl_core_lag1"] = lag_lookup(core, panel["month"], 1) - lag_lookup(core, panel["month"], 2)
    add_nse_return(panel, nse, 6, monthly=True)
    return finish(panel, "H3 monthly", "panel_H3_monthly.csv",
                  need=["bank_ret", "dinfl_head_lag1", "nse20_ret_pct"],
                  spot_cols=["period_end", "infl_head_t1", "infl_head_t2", "dinfl_head_lag1", "dinfl_core_lag1",
                             "nse20_ret_pct", "bank_ret"],
                  gap_days=35)


# ------------------------------------------------------------------ H4 (needs merged LONG-format liquidity file)
def build_H4(nse):
    panel = weekly_base()
    td = panel["n_trading_days"].where(panel["n_trading_days"] > 0)
    adtv = (panel["sector_turnover_kes"] / td).where(lambda s: s > 0)
    panel["adtv_kes"] = adtv
    panel["dlog_adtv"] = np.log(adtv) - np.log(adtv.shift(1))
    lq = load_raw(LIQUIDITY)
    log("Liquidity indicators found: " + str(sorted(lq["Indicator"].unique())))
    per_date = lq.groupby("Date")["Indicator"].nunique()
    log(f"Indicators per date: {per_date.value_counts().sort_index().to_dict()} (expect 4 everywhere; fewer = truncated chunk)")
    ir = lq[lq["Indicator"] == LIQ_RATE_NAME].copy()
    if ir.empty:
        found = sorted(lq["Indicator"].dropna().unique())
        raise ValueError(f"Cannot build H4: required indicator {LIQ_RATE_NAME!r} is missing from {LIQUIDITY}. "
                         f"Found {found}; merge the complete liquidity source data first.")
    ir["v"] = num_text(ir["Value"])
    ir["week_monday"] = ir["Date"] - pd.to_timedelta(ir["Date"].dt.dayofweek, unit="D")
    wk = ir.groupby("week_monday")["v"].agg(["mean", "count"])
    if wk.empty:
        raise ValueError(f"Cannot build H4: {LIQ_RATE_NAME!r} has no numeric observations in {LIQUIDITY}.")
    log(f"Interbank rate: {ir['Date'].min().date()} -> {ir['Date'].max().date()}; weekly averages {len(wk)}, "
        f"obs per week min {wk['count'].min()} median {int(wk['count'].median())}")
    panel["ib_t1"] = (panel["week_monday"] - pd.Timedelta(days=7)).map(wk["mean"])
    panel["ib_t2"] = (panel["week_monday"] - pd.Timedelta(days=14)).map(wk["mean"])
    panel["dib_lag1"] = panel["ib_t1"] - panel["ib_t2"]
    add_nse_return(panel, nse, 5)
    return finish(panel, "H4 weekly", "panel_H4_weekly.csv",
                  need=["dlog_adtv", "dib_lag1", "nse20_ret_pct"],
                  spot_cols=["period_end", "n_trading_days", "adtv_kes", "dlog_adtv", "ib_t1", "ib_t2", "dib_lag1", "nse20_ret_pct"],
                  gap_days=10)


# ------------------------------------------------------------------ H5
def build_H5(nse):
    b = load_clean(BANK_FLOWS_M)
    panel = pd.DataFrame({"period_end": b["period_end"], "flow_ratio": pd.to_numeric(b["sector_flow_ratio"], errors="coerce")})
    panel["month"] = pd.PeriodIndex(panel["period_end"], freq="M")
    d = load_raw(DEBT)
    d["amt"] = num_text(d["Amount (KES)"])
    d["m"] = d["Date"].dt.to_period("M")
    wide = d.pivot_table(index="m", columns="Debt_type", values="amt", aggfunc="last")
    log("Debt types: " + str(list(wide.columns)))
    tot_df = pd.DataFrame({"Date": wide.index.to_timestamp(), "total": (wide["External Debt"] + wide["Domestic Debt"]).to_numpy(),
                           "dom": wide["Domestic Debt"].to_numpy()})
    tot = monthly_series(tot_df, "total", "Total public debt")
    dom = monthly_series(tot_df, "dom", "Domestic debt")
    g_tot, g_dom = 100 * (tot / tot.shift(1) - 1), 100 * (dom / dom.shift(1) - 1)
    panel["dg_total_lag2"] = lag_lookup(g_tot, panel["month"], 2)
    panel["dg_total_lag1"] = lag_lookup(g_tot, panel["month"], 1)
    panel["dg_dom_lag2"] = lag_lookup(g_dom, panel["month"], 2)
    panel["debt_t2"] = lag_lookup(tot, panel["month"], 2)
    panel["debt_t3"] = lag_lookup(tot, panel["month"], 3)
    add_nse_return(panel, nse, 6, monthly=True)
    return finish(panel, "H5 monthly", "panel_H5_monthly.csv",
                  need=["flow_ratio", "dg_total_lag2", "nse20_ret_pct"],
                  spot_cols=["period_end", "debt_t3", "debt_t2", "dg_total_lag2", "dg_dom_lag2", "nse20_ret_pct", "flow_ratio"],
                  gap_days=35, end=H5_END)


if __name__ == "__main__":
    nse = load_nse()
    for fn in (build_H2, build_H3, build_H1, build_H5):
        fn(nse)
    if os.path.exists(LIQUIDITY):
        build_H4(nse)
    else:
        log(f"\nH4 SKIPPED: {LIQUIDITY} not found (merge the liquidity chunks first, keyed on Date AND Indicator).")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as fh:
        fh.write("\n".join(lines))
    log(f"\nLog saved to {LOG}")
