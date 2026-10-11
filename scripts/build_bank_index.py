"""
build_bank_index.py  |  Phase 2: our own banking index, bank returns and sector flows

Run after load_equities.py:   python scripts/build_bank_index.py
Reads  data/clean/equities_daily.csv
Writes data/clean/  bank_index_daily.csv
                    bank_returns_weekly.csv   bank_returns_monthly.csv
                    bank_flows_weekly.csv     bank_flows_monthly.csv

Design choices (and why):
  * Equal-weighted index (primary): each bank counts the same each day. Rock gives no
    shares-outstanding data, so market-cap weights are impossible, and a price-weighted
    index would let NCBA (KES ~90 a share) dominate.
  * Turnover-weighted index (robustness): weights = each bank's average traded value over
    the previous 60 trading days (lagged one day, so no look-ahead).
  * "Ex-bank" indices (robustness): the equal-weighted index with one bank left out, for the
    drop-one-bank check in the plan.
  * Price returns only (no dividends in the data). Dividend drops around payment dates add
    noise; state this in the limits section.
  * Daily returns use the last valid price, so a missing price gives NaN, never a fake 0%.
  * Weekly/monthly index returns compound the daily index returns.
  * flow_ratio = net foreign flow / traded value (price x volume) on days where the flow is
    known. Rock gives no foreign buys/sales, so total traded value is the denominator.
  * Flow integrity (Phase 3 patch): bank-days flagged by load_equities.py (volume=price, stale
    volume, zero volume with flow) are removed from flow numerators, flow denominators and traded
    value; prices stay. A day is usable for the sector flow ratio only if >= 3 banks are valid.
    A weekly/monthly ratio is blank when fewer than MIN_FLOW_DAYS usable days remain.
    ADTV (average daily traded value) uses only days when ALL five banks have a valid volume, so
    the bank mix is the same every day. Per-bank turnover and "ex-bank" columns are written so the
    drop-one-bank checks need no extra work.
  * Weeks end on Friday ("W-FRI"); T-bill dates fall on Mondays, so map each Monday to the
    Friday of the same week when merging.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/clean"
MIN_BANKS = 3      # need at least 3 of 5 banks priced to publish an index return that day
MIN_FLOW_DAYS = 3  # a weekly/monthly flow ratio or ADTV needs at least this many usable days (change-log item 7)
TW_WINDOW = 60     # trading days for turnover weights


def month_rule():
    try:
        pd.Series([1], index=pd.to_datetime(["2020-01-31"])).resample("ME")
        return "ME"
    except ValueError:
        return "M"


MONTH = month_rule()


def compound(r, rule):
    """Compound returns within each period; NaN if the period has no observations."""
    return np.expm1(np.log1p(r).resample(rule).sum(min_count=1))


def trim_partial(out, ndays, minimum=3):
    """Drop partial periods at the start and end of the sample (fewer than 3 trading days)."""
    ndays = ndays.reindex(out.index)
    while len(out) and ndays.iloc[0] < minimum:
        out, ndays = out.iloc[1:], ndays.iloc[1:]
    while len(out) and ndays.iloc[-1] < minimum:
        out, ndays = out.iloc[:-1], ndays.iloc[:-1]
    return out


def main():
    df = pd.read_csv(CLEAN / "equities_daily.csv", parse_dates=["date"])
    price = df.pivot(index="date", columns="bank", values="price_clean").sort_index()
    turn = df.pivot(index="date", columns="bank", values="turnover_kes").sort_index()
    flow = df.pivot(index="date", columns="bank", values="net_flow_kes").sort_index()
    banks = list(price.columns)

    ret = price / price.ffill().shift(1) - 1
    n = ret.notna().sum(axis=1)

    ew = ret.mean(axis=1)
    ew[n < MIN_BANKS] = np.nan

    w = turn.rolling(TW_WINDOW, min_periods=20).mean().shift(1).where(ret.notna())
    tw = (ret * w).sum(axis=1) / w.sum(axis=1).where(w.sum(axis=1) > 0)
    tw[n < MIN_BANKS] = np.nan

    daily = pd.DataFrame({"ew_ret": ew, "tw_ret": tw, "n_banks": n})
    daily["ew_level"] = 100 * (1 + ew.fillna(0)).cumprod()
    daily["tw_level"] = 100 * (1 + tw.fillna(0)).cumprod()
    for b in banks:
        others = ret.drop(columns=b)
        ex = others.mean(axis=1)
        ex[others.notna().sum(axis=1) < MIN_BANKS] = np.nan
        daily[f"ret_ex_{b}"] = ex
    for b in banks:
        daily[f"ret_{b}"] = ret[b]
    daily.index.name = "date"
    daily.to_csv(CLEAN / "bank_index_daily.csv")

    # weekly / monthly returns
    ret_cols = daily[[c for c in daily.columns if c.startswith(("ret_", "ew_ret", "tw_ret"))]]
    for rule, name in (("W-FRI", "weekly"), (MONTH, "monthly")):
        nd = price.notna().any(axis=1).resample(rule).sum()
        out = compound(ret_cols, rule)
        out["n_trading_days"] = nd
        out = trim_partial(out, nd)
        out.index.name = "period_end"
        out.to_csv(CLEAN / f"bank_returns_{name}.csv")

    # ---- weekly / monthly flows (with the integrity rules) ----
    need_cols = {"flow_exclude", "net_flow_capped_kes"}
    if not need_cols <= set(df.columns):
        raise SystemExit("equities_daily.csv lacks the flow-integrity columns: re-run scripts/load_equities.py first")

    def wide(col):
        return df.pivot(index="date", columns="bank", values=col).reindex(price.index)

    bad = wide("flow_exclude").fillna(False).astype(bool)
    flow_c, turn_c, cap_c = flow.where(~bad), turn.where(~bad), wide("net_flow_capped_kes").where(~bad)
    ok = flow_c.notna() & turn_c.notna()                      # bank-day usable for flow ratios
    day_ok = ok.sum(axis=1) >= MIN_BANKS                      # sector-day usable
    use = ok & pd.DataFrame({b: day_ok for b in banks})
    flow_u, turn_u, cap_u = flow_c.where(use), turn_c.where(use), cap_c.where(use)
    # ADTV: only days when every bank has a valid volume (same bank mix each day)
    adtv_day = turn_c.notna().all(axis=1)
    tv_u = turn_c.where(pd.DataFrame({b: adtv_day for b in banks}))

    for rule, name in (("W-FRI", "weekly"), (MONTH, "monthly")):
        fl = flow_u.resample(rule).sum(min_count=1)
        tu = turn_u.resample(rule).sum(min_count=1)
        cp = cap_u.resample(rule).sum(min_count=1)
        nb = use.resample(rule).sum()                         # usable days per bank
        n_flow = day_ok.resample(rule).sum()                  # usable sector days
        enough = n_flow >= MIN_FLOW_DAYS
        out = pd.DataFrame(index=fl.index)
        for b in banks:
            out[f"net_flow_kes_{b}"] = fl[b]
        for b in banks:
            out[f"flow_ratio_{b}"] = (fl[b] / tu[b]).where(nb[b] >= MIN_FLOW_DAYS)
        for b in banks:
            out[f"turnover_kes_{b}"] = tu[b]
        s_net, s_tu, s_cp = fl.sum(axis=1, min_count=1), tu.sum(axis=1, min_count=1), cp.sum(axis=1, min_count=1)
        out["sector_net_flow_kes"] = s_net
        out["sector_turnover_kes"] = s_tu
        out["sector_flow_ratio"] = (s_net / s_tu).where(enough & (s_tu > 0))
        out["sector_flow_ratio_capped"] = (s_cp / s_tu).where(enough & (s_tu > 0))
        for b in banks:
            den = s_tu - tu[b].fillna(0)
            out[f"sector_flow_ratio_ex_{b}"] = ((s_net - fl[b].fillna(0)) / den).where(enough & (den > 0))
        out["n_flow_days"] = n_flow
        out["n_trading_days"] = price.notna().any(axis=1).resample(rule).sum()
        # average daily traded value (H4 outcome), sector and leave-one-bank-out
        tv = tv_u.resample(rule).sum(min_count=1)
        n_ad = adtv_day.resample(rule).sum()
        ok_ad = n_ad >= MIN_FLOW_DAYS
        tv_s = tv.sum(axis=1, min_count=1)
        out["adtv_kes"] = (tv_s / n_ad).where(ok_ad & (tv_s > 0))
        out["n_adtv_days"] = n_ad
        for b in banks:
            out[f"adtv_ex_{b}"] = ((tv_s - tv[b].fillna(0)) / n_ad).where(ok_ad)
        out = trim_partial(out, out["n_trading_days"])
        out.index.name = "period_end"
        out.to_csv(CLEAN / f"bank_flows_{name}.csv")

    wk = pd.read_csv(CLEAN / "bank_returns_weekly.csv", parse_dates=["period_end"])
    fw = pd.read_csv(CLEAN / "bank_flows_weekly.csv")
    print(f"Daily index: {daily.index.min().date()} to {daily.index.max().date()}, "
          f"{int(daily.ew_ret.notna().sum()):,} index days (banks priced per day: "
          f"mean {n.mean():.2f}; the first day has no prior price, so no return)")
    print(f"Equal-weight index level: 100 -> {daily.ew_level.iloc[-1]:.1f}; "
          f"turnover-weighted: 100 -> {daily.tw_level.iloc[-1]:.1f}")
    print(f"Weekly rows: {len(wk)} | monthly rows: {len(pd.read_csv(CLEAN / 'bank_returns_monthly.csv'))}")
    print("Cumulative price change by bank (first to last clean price):")
    for b in banks:
        s = price[b].dropna()
        print(f"  {b:7s} {s.iloc[0]:7.2f} -> {s.iloc[-1]:7.2f}  ({s.iloc[-1] / s.iloc[0] - 1:+.0%})")
    big = int((fw["sector_flow_ratio"].abs() > 1).sum())
    print(f"Weeks with |sector flow ratio| > 1 (traded-value mismatch): {big} of {len(fw)}")
    print(f"Weeks with a blank sector flow ratio (fewer than {MIN_FLOW_DAYS} usable days): "
          f"{int(fw['sector_flow_ratio'].isna().sum())}  {fw.loc[fw['sector_flow_ratio'].isna(), 'period_end'].tolist()}")
    print(f"Weeks with a blank ADTV: {int(fw['adtv_kes'].isna().sum())}")
    print("Saved 5 files in", CLEAN)


if __name__ == "__main__":
    main()
