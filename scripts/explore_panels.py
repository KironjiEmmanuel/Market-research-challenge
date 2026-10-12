# scripts/explore_panels.py   (Phase 3 v2, steps 1-4: check, plot, big flow weeks, stationarity)
# Run after build_all_panels.py (v3):  python scripts/explore_panels.py   (paths are relative to the repo root)
#
# Writes to outputs/phase3/ :
#   preflight_checks.txt, outliers.csv, top5_flow_weeks.csv, stationarity.csv,
#   plots_<panel>.png, and a log of everything printed.
#
# NO driver-vs-outcome relationships are computed here (that is step 5, a separate script).
# This script only asks: is each table correct, what does each series look like on its own,
# and is each series stationary?

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# ============================ PATHS (repo-relative; run from repo root) ============================
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
CLEAN = str(ROOT / "data/clean")
OUT = str(ROOT / "outputs/phase3")
BANKS = ["Absa", "Co-op", "Equity", "KCB", "NCBA"]

# model variables per panel (the series that actually enter a regression)
PANELS = {
    "H1_weekly":  dict(file="panel_H1_weekly.csv",  freq="W",
                       vars=["flow_ratio", "flow_ratio_eqw", "flow_ratio_capped", "dfx_lag1_pct", "nse20_ret_pct"]),
    "H1_monthly": dict(file="panel_H1_monthly.csv", freq="M",
                       vars=["flow_ratio", "dfx_lag1_pct", "nse20_ret_pct"]),
    "H2_weekly":  dict(file="panel_H2_weekly.csv",  freq="W",
                       vars=["flow_ratio", "flow_ratio_eqw", "flow_ratio_capped", "dy91_lag1",
                             "dspread91m364_lag1", "nse20_ret_pct"]),
    "H3_monthly": dict(file="panel_H3_monthly.csv", freq="M",
                       vars=["bank_ret", "dinfl_head_lag1", "dinfl_core_lag1", "nse20_ret_pct"]),
    "H4_weekly":  dict(file="panel_H4_weekly.csv",  freq="W",
                       vars=["dlog_adtv", "dib_lag1", "nse20_ret_pct"]),
    "H4_monthly": dict(file="panel_H4_monthly.csv", freq="M",
                       vars=["dlog_adtv", "dib_lag1", "nse20_ret_pct"]),
    "H5_monthly": dict(file="panel_H5_monthly.csv", freq="M",
                       vars=["flow_ratio", "flow_ratio_eqw", "flow_ratio_capped", "dg_total_lag3",
                             "dg_total_lag2", "dg_dom_lag3", "nse20_ret_pct"]),
}
HELPER_HINTS = ("_px_", "n_trading", "week_monday", "split_part", "period_end", "month", "turnover", "_t1", "_t2", "_t3",
                "fx_end", "debt_t")
# ================================================================================================

os.makedirs(OUT, exist_ok=True)
LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def load_panel(cfg):
    path = os.path.join(CLEAN, cfg["file"])
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["period_end"] = pd.to_datetime(df["period_end"])
    return df.sort_values("period_end").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# STEP 1a. PRE-FLIGHT CHECKS
# WHY: every later number rests on these tables. A shifted lag or a blank row would quietly
# corrupt every test, so we recompute the key columns from their own helper columns first.
# ----------------------------------------------------------------------------------------------
def preflight(name, df, cfg):
    log(f"\n--- {name} ---")
    log(f"rows={len(df)}  range={df.period_end.min().date()} -> {df.period_end.max().date()}")
    dup = df.period_end.duplicated().sum()
    gaps = df.period_end.diff().dt.days
    limit = 10 if cfg["freq"] == "W" else 35
    log(f"duplicate dates={dup}   gaps > {limit} days={int((gaps > limit).sum())}")
    if cfg["freq"] == "W":
        log(f"weekday of period_end (4=Fri): {df.period_end.dt.weekday.value_counts().to_dict()}")
    nan = df.isna().sum()
    nan = nan[nan > 0]
    log(f"blank cells: {nan.to_dict() if len(nan) else 'none'}")
    if "split_part" in df.columns:
        log(f"split_part counts: {df.split_part.value_counts().sort_index().to_dict()}")

    # recompute checks (each prints PASS/FAIL)
    if {"nse20_px_end", "nse20_px_prev", "nse20_ret_pct"} <= set(df.columns):
        calc = (df.nse20_px_end / df.nse20_px_prev - 1) * 100
        ok = np.allclose(calc, df.nse20_ret_pct, atol=1e-6)
        log(f"recompute nse20_ret_pct from prices: {'PASS' if ok else 'FAIL'}")
    if {"y91_t1", "y91_t2", "dy91_lag1"} <= set(df.columns):
        ok = np.allclose(df.y91_t1 - df.y91_t2, df.dy91_lag1, atol=1e-9)
        log(f"recompute dy91_lag1 = y91_t1 - y91_t2 (t1 = latest known Monday, t2 = the one before): {'PASS' if ok else 'FAIL'}")
        # lag logic: t2 must be a Monday strictly before the outcome week starts
        if "week_monday" in df.columns:
            log("  (lag timing is checked by eye below: y91 window dates vs week_monday)")
    if {"fx_chg_pct", "dfx_lag1_pct"} <= set(df.columns):
        shifted = df.fx_chg_pct.shift(1)
        m = shifted.notna()
        ok = np.allclose(shifted[m], df.dfx_lag1_pct[m], atol=1e-6)
        log(f"dfx_lag1_pct equals previous row's fx_chg_pct: {'PASS' if ok else 'FAIL'}")
        zero_share = (df.dfx_lag1_pct.abs() < 1e-9).mean()
        log(f"share of weeks with EXACTLY zero lagged FX change: {zero_share:.1%}")
        log(f"weeks with |lagged FX change| > 0.2%: {(df.dfx_lag1_pct.abs() > 0.2).sum()}")
    if {"dg_total_lag3", "debt_t3", "debt_t4"} <= set(df.columns):
        calc = (df.debt_t3 / df.debt_t4 - 1) * 100
        ok = np.allclose(calc, df.dg_total_lag3, atol=1e-6)
        log(f"recompute dg_total_lag3 = debt_t3/debt_t4 - 1 (t3 = value 3 months before the outcome month): {'PASS' if ok else 'FAIL'}")
    if {"y91_t1", "y91_t2", "y364_t1", "y364_t2", "dspread91m364_lag1"} <= set(df.columns):
        calc = (df.y91_t1 - df.y364_t1) - (df.y91_t2 - df.y364_t2)
        m = calc.notna() & df.dspread91m364_lag1.notna()
        ok = np.allclose(calc[m], df.dspread91m364_lag1[m], atol=1e-9) and (calc.isna() == df.dspread91m364_lag1.isna()).all()
        log(f"recompute dspread91m364_lag1 = change in (91-day minus 364-day): {'PASS' if ok else 'FAIL'} "
            f"({int(df.dspread91m364_lag1.isna().sum())} blank rows, from the masked source fault)")
    for c in ("flow_ratio", "flow_ratio_eqw", "flow_ratio_capped"):
        if c in df.columns:
            log(f"{c} within [-1, 1]: {'PASS' if df[c].abs().max() <= 1 else 'FAIL (max ' + str(round(df[c].abs().max(), 2)) + ')'}")
    ex = [c for c in df.columns if "_ex_" in c]
    if ex:
        log(f"drop-one-bank columns present ({len(ex)}): {'PASS' if len(ex) == 5 else 'FAIL'}; blanks in them: {int(df[ex].isna().sum().sum())}")
    if {"infl_head_t1", "infl_head_t2", "dinfl_head_lag1"} <= set(df.columns):
        ok = np.allclose(df.infl_head_t1 - df.infl_head_t2, df.dinfl_head_lag1, atol=1e-9)
        log(f"recompute dinfl_head_lag1 = t1 - t2: {'PASS' if ok else 'FAIL'}")

    # show the first, a middle and the last row of the lag window so the user can verify by hand
    show = [c for c in ["period_end", "week_monday", "y91_t1", "y91_t2", "dy91_lag1",
                        "infl_head_t1", "infl_head_t2", "debt_t4", "debt_t3", "ib_t1", "ib_t2"] if c in df.columns]
    if len(show) > 2:
        idx = [0, len(df) // 2, len(df) - 1]
        log("lag-window rows to verify by hand against the raw file:")
        log(df.loc[idx, show].to_string(index=False))


# ----------------------------------------------------------------------------------------------
# STEP 2. PLOTS
# WHY: a regression only sees numbers. Plotting shows structural breaks, flat stretches (FX!),
# and single-point outliers that would otherwise dominate a fitted line.
# ----------------------------------------------------------------------------------------------
def plot_panel(name, df, vars_):
    n = len(vars_)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.3 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, vars_):
        ax.plot(df.period_end, df[v], lw=0.9)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel(v, fontsize=8)
    axes[0].set_title(f"{name}: every model variable over time")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"plots_{name}.png"), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# STEP 2b. OUTLIERS (robust z-score: median and MAD, so the outlier does not hide itself)
# ----------------------------------------------------------------------------------------------
def find_outliers(name, df, vars_):
    rows = []
    for v in vars_:
        x = df[v].astype(float)
        med = x.median()
        mad = (x - med).abs().median()
        if mad == 0:
            continue
        rz = 0.6745 * (x - med) / mad
        for i in rz.index[rz.abs() > 4]:
            rows.append(dict(panel=name, variable=v, period_end=df.period_end[i].date(),
                             value=x[i], robust_z=round(rz[i], 1)))
    return rows


# ----------------------------------------------------------------------------------------------
# STEP 3. TOP-5 FLOW WEEKS PER BANK
# WHY: foreign flows are lumpy. A few huge weeks (block trades, index events, one investor
# exiting) can drive any result. We list them so you can research what each was and decide
# if it is a real market event (keep) or a data problem (fix).
# ----------------------------------------------------------------------------------------------
def flow_integrity():
    """WHY: net foreign flow = buys - sells, and each is part of traded value, so |net flow| can never
    exceed turnover. load_equities.py now applies three pattern rules (volume=price, stale volume,
    zero volume with flow) and excludes those bank-days from all flow numerators and denominators.
    This report shows what was excluded and what still breaks the bound (Type C, kept, capped copy exists)."""
    path = os.path.join(CLEAN, "equities_daily.csv")
    if not os.path.exists(path):
        return
    e = pd.read_csv(path, parse_dates=["date"])
    if "flow_exclude" not in e.columns:
        log("\nFLOW INTEGRITY: equities_daily.csv has no integrity columns; re-run load_equities.py")
        return
    ex = e[e.flow_exclude]
    log(f"\nFLOW INTEGRITY: {len(ex)} bank-days excluded (volume=price {int(e.flag_vol_eq_price.sum())}, "
        f"stale volume {int(e.flag_stale_vol.sum())}, zero volume with flow {int(e.flag_zero_vol_flow.sum())})")
    log("excluded days by date: " + str(ex.groupby(ex.date.dt.date).size().to_dict()))
    c = e[e.flag_flow_gt_turnover]
    log(f"TYPE C (|flow| > turnover, volume plausible; KEPT in primary, capped in the sensitivity): {len(c)} bank-days")
    log(c[["date", "bank", "price", "volume", "net_flow_kes", "turnover_kes"]].to_string(index=False))
    c.to_csv(os.path.join(OUT, "flow_typeC_flags.csv"), index=False)


def top_flow_weeks():
    path = os.path.join(CLEAN, "bank_flows_weekly.csv")
    if not os.path.exists(path):
        log("bank_flows_weekly.csv not found; skipping step 3")
        return
    f = pd.read_csv(path)
    f["period_end"] = pd.to_datetime(f["period_end"])
    out = []
    for b in BANKS + ["SECTOR"]:
        ncol = "sector_net_flow_kes" if b == "SECTOR" else f"net_flow_kes_{b}"
        rcol = "sector_flow_ratio" if b == "SECTOR" else f"flow_ratio_{b}"
        t = f.assign(abs_flow=f[ncol].abs()).nlargest(5, "abs_flow")
        for _, r in t.iterrows():
            out.append(dict(bank=b, week_ending=r.period_end.date(),
                            net_flow_kes_m=round(r[ncol] / 1e6, 1), flow_ratio=round(r[rcol], 3),
                            what_was_it="(fill in after research)"))
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(OUT, "top5_flow_weeks.csv"), index=False)
    log("\nTOP-5 BIGGEST FLOW WEEKS (absolute net flow, KES million); research each one:")
    log(res.drop(columns="what_was_it").to_string(index=False))


# ----------------------------------------------------------------------------------------------
# STEP 4. STATIONARITY
# ADF null = "series has a unit root (non-stationary)". Small p => reject => stationary.
# KPSS null is the OPPOSITE ("stationary"). Using both guards against ADF's low power on short
# samples: we trust a series when ADF rejects AND KPSS does not.
# ----------------------------------------------------------------------------------------------
def stationarity(name, df, vars_):
    rows = []
    for v in vars_:
        x = df[v].astype(float).dropna().values
        if np.std(x) == 0 or len(x) < 20:
            rows.append(dict(panel=name, variable=v, n=len(x), note="constant or too short"))
            continue
        # short samples: cap the lag search (AIC on n<60 picks too many lags and destroys power)
        a = adfuller(x, regression="c", autolag="AIC", maxlag=3 if len(x) < 60 else None)
        try:
            k = kpss(x, regression="c", nlags="auto")
            kp = k[1]
        except Exception:
            kp = np.nan
        adf_ok = a[1] < 0.05
        kpss_ok = (kp >= 0.05) if not np.isnan(kp) else None
        if adf_ok and kpss_ok:
            call = "stationary"
        elif adf_ok and kpss_ok is False:
            call = "mixed: check plot"
        elif (not adf_ok) and kpss_ok is False:
            call = "NON-stationary: difference"
        else:
            call = "unclear (low power?)"
        rows.append(dict(panel=name, variable=v, n=len(x), adf_stat=round(a[0], 2), adf_p=round(a[1], 4),
                         adf_lags=a[2], kpss_p=round(kp, 3) if not np.isnan(kp) else np.nan, call=call))
    return rows


if __name__ == "__main__":
    log("PHASE 3 (steps 1-4): pre-flight, plots, outliers, big flow weeks, stationarity")
    all_outliers, all_stat = [], []
    for name, cfg in PANELS.items():
        df = load_panel(cfg)
        if df is None:
            log(f"\n--- {name} --- file {cfg['file']} not found: SKIPPED")
            continue
        vars_ = cfg["vars"]
        if vars_ is None:   # auto-detect for the H4 panel
            vars_ = [c for c in df.select_dtypes("number").columns if not any(h in c for h in HELPER_HINTS)]
            log(f"\n{name}: auto-detected model variables -> {vars_}")
        preflight(name, df, cfg)
        plot_panel(name, df, vars_)
        all_outliers += find_outliers(name, df, vars_)
        all_stat += stationarity(name, df, vars_)

    if all_outliers:
        o = pd.DataFrame(all_outliers)
        o.to_csv(os.path.join(OUT, "outliers.csv"), index=False)
        log("\nOUTLIERS (robust z > 4): decide for each: real event or data problem?")
        log(o.to_string(index=False))
    flow_integrity()
    top_flow_weeks()
    s = pd.DataFrame(all_stat)
    s.to_csv(os.path.join(OUT, "stationarity.csv"), index=False)
    log("\nSTATIONARITY TABLE (ADF p<0.05 = stationary; KPSS p>=0.05 = stationary):")
    log(s.to_string(index=False))
    with open(os.path.join(OUT, "phase3_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG))
    log(f"\nSaved outputs to {OUT}/")
