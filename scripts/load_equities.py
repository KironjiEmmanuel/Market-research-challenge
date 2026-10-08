"""
load_equities.py  |  Phase 2: clean loader for the Rock equity export

Run from the project root:   python scripts/load_equities.py
Reads   data/raw/Equity prices data.csv   (never modified)
Writes  data/clean/equities_daily.csv     (one row per bank per weekday, from Sep 2017)

Why a custom loader (pd.read_csv would silently corrupt this file):
  1. Net Foreign Flow has thousands commas with no quotes, so one number spans 1-3 CSV fields.
  2. Inside those groups, leading zeros were dropped: "-65,690,20" is really -65,690,020.00
     (confirmed against the Rock portal for KCB on 29 Jun 2026).
  3. Sunday rows copy the next Monday's volume, so summing every row double counts.
  4. Public-holiday weekdays (no prices at all) also carry a copy of the next trading day's volume.
  5. A few single-day price glitches exist (e.g. NCBA on 19 Mar 2020 shows KES 2,825).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data/raw/Equity prices data.csv"
OUT = Path("C:/Users/KIRONJI/Desktop/Rock research project/data/clean")
START = "2017-09-01"   # start of the analysis window
SPIKE = 0.10           # a one-day move above 10% that fully reverses next day is a glitch
REVIEW = 0.15          # any other one-day move above 15% is kept but listed for review

BANKS = {
    "KCB Group Ltd Ord 1.00": "KCB",
    "Absa Bank Kenya Plc Ord 0.50": "Absa",
    "NCBA Bank Ltd Ord 5.00": "NCBA",
    "Equity Group Holdings Ltd Ord 0.50": "Equity",
    "The Co-operative Bank of Kenya Ltd Ord 1.00": "Co-op",
}
PCT = re.compile(r"^[+-]?\d+(\.\d+)?%$")


def parse_flow(tokens):
    """['-65','690','20'] -> (-65690020.0, False). Every group after the first must hold
    3 integer digits, so short groups get their leading zeros back. The flag is True when
    the pieces cannot be a valid thousands-grouped number."""
    if not tokens or tokens == ["-"]:
        return np.nan, False
    flag = len(tokens[0].lstrip("-").split(".")[0]) > 3
    text = tokens[0]
    for tk in tokens[1:]:
        whole, dot, dec = tk.partition(".")
        flag = flag or len(whole) > 3
        text += whole.zfill(3) + dot + dec
    try:
        return float(text), flag
    except ValueError:
        return np.nan, True


def read_raw(path):
    """Split each line by hand: date, security, price, volume, <flow pieces>, change %."""
    text = path.read_text(encoding="utf-8-sig").replace("\r", "")
    recs, problems = [], []
    for n, line in enumerate(text.split("\n")[1:], start=2):
        if not line.strip():
            continue
        t = line.rstrip(", ").split(",")
        if len(t) < 5 or not (t[-1] == "-" or PCT.match(t[-1])):
            problems.append((n, line))
            continue
        recs.append(dict(date=t[0], security=t[1], price=t[2], volume=t[3],
                         flow_tokens=t[4:-1], chg=t[-1]))
    return recs, problems


def remove_spikes(s):
    """Blank one-day price spikes that revert next day. Returns (cleaned, removed, review).
    'review' = big moves that did NOT revert (could be real news), kept but listed."""
    s = s.copy()
    removed = []
    for _ in range(5):
        p = s.dropna()
        prev, nxt = p.shift(1), p.shift(-1)
        spike = ((p / prev - 1).abs() > SPIKE) & ((nxt / prev - 1).abs() < SPIKE / 3)
        if not spike.any():
            break
        removed += list(p.index[spike])
        s[p.index[spike]] = np.nan
    p = s.dropna()
    review = list(p.index[(p / p.shift(1) - 1).abs() > REVIEW])
    return s, removed, review


def main():
    recs, problems = read_raw(RAW)
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["bank"] = df["security"].map(BANKS)
    unknown = df.loc[df.bank.isna(), "security"].unique()
    df = df.dropna(subset=["bank"]).copy()
    if df.duplicated(["date", "bank"]).any():
        raise SystemExit("Duplicate (date, bank) rows found: investigate before continuing")

    df["price"] = pd.to_numeric(df["price"].str.replace("KES", "", regex=False)
                                .str.replace(",", "", regex=False).str.strip(), errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"].str.replace(",", "", regex=False), errors="coerce")
    parsed = df["flow_tokens"].apply(parse_flow)
    df["net_flow_kes"] = [p[0] for p in parsed]
    df["flow_decode_flag"] = [p[1] for p in parsed]
    df["chg_pct_reported"] = pd.to_numeric(
        df["chg"].str.replace("%", "", regex=False).str.replace("+", "", regex=False), errors="coerce")

    # Sunday rows should copy the next Monday's volume. Verify before dropping them.
    vol = df.pivot(index="date", columns="bank", values="volume")
    same = tot = 0
    for d in vol.index[vol.index.dayofweek == 6]:
        nxt = d + pd.Timedelta(days=1)
        if nxt in vol.index:
            both = vol.loc[d].notna() & vol.loc[nxt].notna()
            tot += int(both.sum())
            same += int((vol.loc[d][both] == vol.loc[nxt][both]).sum())
    weekend_rows = int((df.date.dt.dayofweek >= 5).sum())

    df = df[(df.date.dt.dayofweek < 5) & (df.date >= START)].sort_values(["bank", "date"]).copy()

    # Weekdays where no bank has a price are non-trading days (holidays). Their volume rows
    # mostly copy the next trading day, so drop them entirely.
    px = df.pivot(index="date", columns="bank", values="price")
    vol_all = df.pivot(index="date", columns="bank", values="volume")
    closed = px.index[px.isna().all(axis=1)]
    nxt_same = 0
    trading = px.index.difference(closed)
    for d in closed:
        later = trading[trading > d]
        if len(later) and vol_all.loc[d].notna().any():
            both = vol_all.loc[d].notna() & vol_all.loc[later[0]].notna()
            nxt_same += int(both.any() and (vol_all.loc[d][both] == vol_all.loc[later[0]][both]).all())
    closed_with_vol = int(vol_all.loc[closed].notna().any(axis=1).sum())
    df = df[~df.date.isin(closed)].copy()

    # price cleaning per bank
    px = df.pivot(index="date", columns="bank", values="price")
    clean, removed, review = {}, {}, {}
    for b in px.columns:
        clean[b], removed[b], review[b] = remove_spikes(px[b])
    clean = pd.DataFrame(clean)
    df["price_clean"] = [clean.at[d, b] for d, b in zip(df.date, df.bank)]
    last_px = clean.ffill()
    df["turnover_kes"] = df["volume"] * [
        pc if pd.notna(pc) else last_px.at[d, b]
        for d, b, pc in zip(df.date, df.bank, df.price_clean)]
    df["no_price_day"] = df["price_clean"].isna()

    cols = ["date", "bank", "price", "price_clean", "volume", "net_flow_kes",
            "turnover_kes", "chg_pct_reported", "flow_decode_flag", "no_price_day"]
    OUT.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT / "equities_daily.csv", index=False)

    # ---- audit printout ----
    print(f"Read {len(recs):,} rows from {RAW.name} ({len(problems)} unparseable lines, "
          f"{len(unknown)} unknown securities)")
    print(f"Sunday volume equals next Monday's: {same}/{tot}; weekend rows dropped: {weekend_rows:,}")
    print(f"Kept {len(df):,} bank-weekday rows, {df.date.min().date()} to {df.date.max().date()}, "
          f"{df.date.nunique():,} weekdays")
    print(f"Flow decode: missing ('-') {df.net_flow_kes.isna().sum():,} | "
          f"flagged as undecodable {int(df.flow_decode_flag.sum())}")
    print(f"Non-trading weekdays dropped: {len(closed)} ({closed_with_vol} carried volume; "
          f"{nxt_same} of those equal the next trading day's volume)")
    print(f"Bank-days with no price on a trading day (no trades): {int(df.no_price_day.sum())}")
    for b in px.columns:
        print(f"  {b:7s} spikes removed: {[str(d.date()) for d in removed[b]]} | "
              f"big moves kept for review: {[str(d.date()) for d in review[b]]}")
    print(f"Saved: {OUT / 'equities_daily.csv'}")


if __name__ == "__main__":
    main()
