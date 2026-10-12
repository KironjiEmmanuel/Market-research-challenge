"""
run_models.py -- Phase 4, Market Research Challenge (Rock Data Plus)

Runs the five frozen test cards ONCE, as written in docs/test_cards.md v1.2.
No specification search: everything is driven by the CARDS config table below.

Run from the repo root:   python scripts/run_models.py
Reads   data/clean/panel_*.csv
Writes  outputs/phase4/  (run_log.txt, scorecard.csv, results_detail.csv,
                          granger.csv, p12_percentiles.csv, ols_summary_H*.txt)

Safety: refuses to run if docs/test_cards.md is untracked or modified in git,
because the cards must be committed (frozen) before the first model run.
Use --skip-freeze-check only for a dry run on test data.
"""
import contextlib
import io
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs" / "phase4"

Z90 = 1.645      # two-sided 90% interval
P_CUT = 0.10     # significance threshold in the verdict rules
TIGHT = 0.20     # "tight interval" half-width in standardized units (change log 15)
SCALE = 100.0    # decimal outcomes -> points (flow_ratio, bank_ret, dlog_adtv)


# --------------------------------------------------------------------------
# Config table: one entry per card. Mirrors docs/test_cards.md v1.2.
# --------------------------------------------------------------------------
CARDS = {
    "H1": dict(
        title="Currency risk", panel="panel_H1_weekly.csv",
        y="flow_ratio", x="dfx_lag1_pct", ctrl="nse20_ret_pct",
        maxlags=4, lb=8, expected=-1, ex_prefix="flow_ratio_ex_",
        x_unit="% weekly change in USD/KES",
        y_unit="points of flow ratio",
        check2=dict(kind="panel", panel="panel_H1_monthly.csv", maxlags=3,
                    label="monthly frequency, 1-month lag"),
        alt_y=["flow_ratio_eqw", "flow_ratio_capped"], alt_x=[],
        drop_rows=[("without rows 2024-02-23 and 2024-03-22", ["2024-02-23", "2024-03-22"]),
                   ("without week ending 2025-09-19", ["2025-09-19"])],
        granger=dict(drv="fx_chg_pct", shift=0),
        p12=dict(col="dfx_lag1_pct", kind="compound", window=52, unit="% change in USD/KES"),
    ),
    "H2": dict(
        title="Competition for money", panel="panel_H2_weekly.csv",
        y="flow_ratio", x="dy91_lag1", ctrl="nse20_ret_pct",
        maxlags=4, lb=8, expected=-1, ex_prefix="flow_ratio_ex_",
        x_unit="pp weekly change in 91-day yield",
        y_unit="points of flow ratio",
        check2=dict(kind="driver", x="dspread91m364_lag1",
                    label="swap driver to change in (91d minus 364d) spread"),
        alt_y=["flow_ratio_eqw", "flow_ratio_capped"], alt_x=[],
        drop_rows=[("without week ending 2025-09-19", ["2025-09-19"])],
        granger=dict(drv="dy91_lag1", shift=-1),
        p12=dict(col="dy91_lag1", kind="sum", window=52, unit="pp change in 91-day yield"),
    ),
    "H3": dict(
        title="Real returns", panel="panel_H3_monthly.csv",
        y="bank_ret", x="dinfl_head_lag1", ctrl="nse20_ret_pct",
        maxlags=3, lb=6, expected=-1, ex_prefix="ret_ex_",
        x_unit="pp monthly change in headline inflation",
        y_unit="percentage points of monthly bank return",
        check2=dict(kind="driver", x="dinfl_core_lag1",
                    label="swap driver to change in core inflation"),
        alt_y=[], alt_x=[], drop_rows=[], granger=None,
        p12=dict(col="dinfl_head_lag1", kind="sum", window=12, unit="pp change in headline inflation"),
    ),
    "H4": dict(
        title="Funding conditions", panel="panel_H4_weekly.csv",
        y="dlog_adtv", x="dib_lag1", ctrl="nse20_ret_pct",
        maxlags=4, lb=8, expected=-1, ex_prefix="dlog_adtv_ex_",
        x_unit="pp weekly change in interbank rate",
        y_unit="log-points (~%) of avg daily traded value",
        check2=dict(kind="panel", panel="panel_H4_monthly.csv", maxlags=3,
                    label="monthly frequency, 1-month lag"),
        alt_y=[], alt_x=[], drop_rows=[], granger=None,
        p12=dict(col="dib_lag1", kind="sum", window=52, unit="pp change in interbank rate"),
    ),
    "H5": dict(
        title="Sovereign risk", panel="panel_H5_monthly.csv",
        y="flow_ratio", x="dg_total_lag3", ctrl="nse20_ret_pct",
        maxlags=3, lb=6, expected=-1, ex_prefix="flow_ratio_ex_",
        x_unit="% monthly growth in total public debt (lag 3)",
        y_unit="points of flow ratio",
        check2=dict(kind="driver", x="dg_dom_lag3",
                    label="swap driver to DOMESTIC debt growth, lag 3"),
        alt_y=["flow_ratio_eqw", "flow_ratio_capped"],
        alt_x=[("lag 2 months (dg_total_lag2)", "dg_total_lag2")],
        drop_rows=[], granger=None,
        p12=dict(col="dg_total_lag3", kind="compound", window=12, unit="% 12-month debt growth"),
    ),
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def freeze_check(skip):
    if skip:
        print("FREEZE CHECK SKIPPED (dry run). Do not report these numbers.")
        return
    try:
        dirty = git("status", "--porcelain", "--", "docs/test_cards.md")
        head = git("rev-parse", "--short", "HEAD")
    except Exception as exc:  # git missing, not a repo, or no commits
        sys.exit(f"STOP: could not verify the card freeze with git ({exc}). Commit the cards first.")
    if dirty:
        sys.exit("STOP: docs/test_cards.md is untracked or modified. Commit it first (that is the freeze).")
    print(f"Freeze check OK. Cards committed; HEAD = {head}")


def load(name):
    df = pd.read_csv(CLEAN / name)
    df["period_end"] = pd.to_datetime(df["period_end"])
    return df.sort_values("period_end").reset_index(drop=True)


def sign(v):
    return int(np.sign(v))


def fit(d, y, x, ctrl, maxlags):
    """OLS of y on x (+ ctrl) with Newey-West (HAC) errors. Rows with any NaN are dropped."""
    need = [c for c in (y, x, ctrl) if c]
    d = d.dropna(subset=need).sort_values("period_end")
    X = sm.add_constant(d[[x] + ([ctrl] if ctrl else [])].astype(float))
    m = sm.OLS(d[y].astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    b, se, p = float(m.params[x]), float(m.bse[x]), float(m.pvalues[x])
    sdx, sdy = float(d[x].std(ddof=1)), float(d[y].std(ddof=1))
    k = sdx / sdy
    return dict(n=int(m.nobs), coef=b, se=se, p=p, lo=b - Z90 * se, hi=b + Z90 * se,
                sdx=sdx, sdy=sdy, z=b * k, z_lo=(b - Z90 * se) * k, z_hi=(b + Z90 * se) * k,
                r2=float(m.rsquared), model=m)


def detail_row(card, group, run, r, psign):
    return dict(card=card, group=group, run=run, n=r["n"], coef=r["coef"], se=r["se"], p=r["p"],
                lo90=r["lo"], hi90=r["hi"], z=r["z"], sign=sign(r["coef"]),
                same_sign_as_primary=(sign(r["coef"]) == psign))


def verdict(prim, expected, n_pass):
    s = sign(prim["coef"])
    if prim["p"] < P_CUT:
        if s == expected and n_pass >= 2:
            return "Supported"
        if s == -expected and n_pass >= 2:
            return "Opposite sign"
        return "Inconclusive"
    if prim["z_lo"] > -TIGHT and prim["z_hi"] < TIGHT:
        return "Not supported"
    return "Inconclusive"


def plain_words(cfg, r):
    f = SCALE
    return (f"1 SD of driver = {r['sdx']:.3g} {cfg['x_unit']} -> {r['coef'] * r['sdx'] * f:+.2f} "
            f"{cfg['y_unit']} (90% CI {r['lo'] * r['sdx'] * f:+.2f} to {r['hi'] * r['sdx'] * f:+.2f}); "
            f"standardized {r['z']:+.2f} SD [{r['z_lo']:+.2f}, {r['z_hi']:+.2f}]")


def granger_p(d, effect, cause, order=2):
    d = d[[effect, cause]].dropna()
    with contextlib.redirect_stdout(io.StringIO()):
        res = grangercausalitytests(d, maxlag=order)
    return float(res[order][0]["ssr_ftest"][1]), len(d)


def run_granger(key, cfg, df):
    g = cfg["granger"]
    d = pd.DataFrame({"period_end": df["period_end"], "flow": df[cfg["y"]]})
    drv = df[g["drv"]]
    if g["shift"] != 0:  # contemporaneous change = next row's lagged change, only if rows are 7 days apart
        gap = (df["period_end"].shift(-1) - df["period_end"]).dt.days
        d["drv"] = drv.shift(-1).where(gap == 7)
    else:
        d["drv"] = drv
    rows = []
    p1, n1 = granger_p(d, "flow", "drv")
    rows.append(dict(card=key, direction="driver -> flows", order=2, p_ssr_F=p1, n=n1))
    p2, n2 = granger_p(d, "drv", "flow")
    rows.append(dict(card=key, direction="flows -> driver", order=2, p_ssr_F=p2, n=n2))
    return rows


def p12_table(key, cfg, df):
    s = cfg["p12"]
    v = df[s["col"]].astype(float).dropna()
    if s["kind"] == "compound":
        roll = (1 + v / 100).rolling(s["window"]).apply(np.prod, raw=True).sub(1).mul(100)
    else:
        roll = v.rolling(s["window"]).sum()
    roll = roll.dropna()
    q = roll.quantile([0.10, 0.50, 0.90])
    return dict(card=key, measure=s["unit"], window_rows=s["window"], n_windows=len(roll),
                p10=q.iloc[0], p50=q.iloc[1], p90=q.iloc[2])


# --------------------------------------------------------------------------
# One card
# --------------------------------------------------------------------------
def run_card(key, cfg):
    print(f"\n{'=' * 18} {key}: {cfg['title']} {'=' * 18}")
    df = load(cfg["panel"])
    y, x, c, L = cfg["y"], cfg["x"], cfg["ctrl"], cfg["maxlags"]

    prim = fit(df, y, x, c, L)
    psign = sign(prim["coef"])
    lb_p = float(acorr_ljungbox(prim["model"].resid, lags=[cfg["lb"]], return_df=True)["lb_pvalue"].iloc[0])
    (OUT / f"ols_summary_{key}.txt").write_text(str(prim["model"].summary()), encoding="utf-8")

    rows = [detail_row(key, "primary", f"{y} ~ {x} + {c}", prim, psign)]
    print(f"Primary: n={prim['n']}  coef={prim['coef']:+.5f}  HAC se={prim['se']:.5f}  p={prim['p']:.4f}  R2={prim['r2']:.3f}")
    print("  " + plain_words(cfg, prim))
    print(f"  Ljung-Box p (lag {cfg['lb']}) on residuals: {lb_p:.3f}" + ("  <-- autocorrelation left" if lb_p < 0.05 else ""))
    if prim["r2"] > 0.5:
        print("  WARNING: R2 > 0.5. Recheck stationarity before trusting this.")

    # ---- Check 1: drop one bank ------------------------------------------
    ex_cols = sorted(col for col in df.columns if col.startswith(cfg["ex_prefix"]))
    if len(ex_cols) != 5:
        print(f"  WARNING: expected 5 drop-one-bank columns, found {len(ex_cols)}: {ex_cols}")
    same = 0
    for col in ex_cols:
        r = fit(df, col, x, c, L)
        rows.append(detail_row(key, "check1_drop_bank", col, r, psign))
        same += sign(r["coef"]) == psign
    c1 = same >= 4
    print(f"  Check 1 drop-one-bank: sign unchanged in {same} of {len(ex_cols)} -> {'pass' if c1 else 'FAIL'}")

    # ---- Check 2: swap driver or frequency -------------------------------
    c2cfg = cfg["check2"]
    if c2cfg["kind"] == "driver":
        r2 = fit(df, y, c2cfg["x"], c, L)
    else:
        r2 = fit(load(c2cfg["panel"]), y, x, c, c2cfg["maxlags"])
    rows.append(detail_row(key, "check2", c2cfg["label"], r2, psign))
    c2 = sign(r2["coef"]) == psign
    print(f"  Check 2 {c2cfg['label']}: coef={r2['coef']:+.5f} p={r2['p']:.3f} n={r2['n']} -> {'pass' if c2 else 'FAIL'}")

    # ---- Check 3: split ---------------------------------------------------
    if "split_part" in df.columns and df["split_part"].nunique() == 2:
        parts = sorted(df["split_part"].unique())
        masks = [df["split_part"] == p for p in parts]
        how = "split_part column"
    else:
        if key == "H1":
            raise RuntimeError("H1 needs the split_part column (split at 22 Jan 2024).")
        med = df["period_end"].sort_values().iloc[len(df) // 2]
        masks = [df["period_end"] < med, df["period_end"] >= med]
        how = "median period"
    c3_signs = []
    for i, mk in enumerate(masks, 1):
        r = fit(df[mk], y, x, c, L)
        rows.append(detail_row(key, "check3_split", f"part {i}", r, psign))
        c3_signs.append(sign(r["coef"]) == psign)
        print(f"  Check 3 part {i} ({how}): n={r['n']} coef={r['coef']:+.5f} p={r['p']:.3f}")
    c3 = all(c3_signs)
    print(f"  Check 3 split -> {'pass' if c3 else 'FAIL'}")

    n_pass = int(c1) + int(c2) + int(c3)
    v = verdict(prim, cfg["expected"], n_pass)
    print(f"  Checks passed: {n_pass} of 3   VERDICT (by rule): {v}   [confidence: your judgement]")

    # ---- Sensitivity rows (reported, never checks) -----------------------
    r = fit(df, y, x, None, L)
    rows.append(detail_row(key, "sensitivity", "without NSE 20 control", r, psign))
    for col in cfg["alt_y"]:
        rows.append(detail_row(key, "sensitivity", f"outcome = {col}", fit(df, col, x, c, L), psign))
    for label, col in cfg["alt_x"]:
        rows.append(detail_row(key, "sensitivity", label, fit(df, y, col, c, L), psign))
    for label, dates in cfg["drop_rows"]:
        keep = df[~df["period_end"].isin(pd.to_datetime(dates))]
        if len(keep) == len(df):
            print(f"  WARNING: sensitivity '{label}' removed no rows (dates not in panel).")
        rows.append(detail_row(key, "sensitivity", label, fit(keep, y, x, c, L), psign))
    print("  Sensitivity (not checks):")
    for rw in rows:
        if rw["group"] == "sensitivity":
            print(f"    {rw['run']:<48} n={rw['n']:<4} coef={rw['coef']:+.5f} p={rw['p']:.3f} sign={'same' if rw['same_sign_as_primary'] else 'DIFFERENT'}")

    # ---- Granger (descriptive, never changes a verdict) ------------------
    g_rows = []
    if cfg["granger"]:
        try:
            g_rows = run_granger(key, cfg, df)
            for gr in g_rows:
                print(f"  Granger {gr['direction']} (order 2): p={gr['p_ssr_F']:.3f} n={gr['n']}")
        except Exception as exc:
            print(f"  Granger failed: {exc}")

    score = dict(
        card=key, hypothesis=cfg["title"], driver=x,
        window=f"{df['period_end'].min():%Y-%m-%d} to {df['period_end'].max():%Y-%m-%d}",
        n=prim["n"], expected="negative", found="negative" if psign < 0 else "positive" if psign > 0 else "zero",
        coef=prim["coef"], p=prim["p"], ci90_lo=prim["lo"], ci90_hi=prim["hi"],
        z=prim["z"], z_lo=prim["z_lo"], z_hi=prim["z_hi"], size_plain=plain_words(cfg, prim),
        check1_drop_bank="pass" if c1 else "FAIL", check2="pass" if c2 else "FAIL",
        check3_split="pass" if c3 else "FAIL", checks_passed=n_pass,
        ljung_box_p=lb_p, verdict=v, confidence="(your judgement)",
    )
    return score, rows, g_rows, p12_table(key, cfg, df)


# --------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "run_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    print("run_models.py -- Phase 4 (one run, no specification search)")
    freeze_check("--skip-freeze-check" in sys.argv)

    scores, details, granger, p12 = [], [], [], []
    failed = []
    for key, cfg in CARDS.items():
        try:
            s, d, g, p = run_card(key, cfg)
            scores.append(s)
            details += d
            granger += g
            p12.append(p)
        except Exception:
            failed.append(key)
            print(f"\n!! {key} FAILED:\n{traceback.format_exc()}")

    pd.DataFrame(scores).to_csv(OUT / "scorecard.csv", index=False)
    pd.DataFrame(details).to_csv(OUT / "results_detail.csv", index=False)
    pd.DataFrame(granger).to_csv(OUT / "granger.csv", index=False)
    pd.DataFrame(p12).to_csv(OUT / "p12_percentiles.csv", index=False)

    print("\n" + "=" * 18 + " SCORECARD " + "=" * 18)
    if scores:
        view = pd.DataFrame(scores)[["card", "window", "n", "found", "p", "z", "z_lo", "z_hi",
                                     "check1_drop_bank", "check2", "check3_split", "verdict"]]
        print(view.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n12-month driver changes (rolling windows, overlapping; approximate):")
    print(pd.DataFrame(p12).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if failed:
        print(f"\nFAILED CARDS: {failed}. Fix the defect and re-run; do not change any card setting.")
    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    main()
