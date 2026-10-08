"""
inventory.py  |  Phase 1: data inventory for the Market Research Challenge

Run from the project root:   python scripts/inventory.py
Reads every CSV under data/raw (read only, never modifies them) and writes
outputs/data_inventory.csv. Paste the printed output back into chat.

Notes from the first review of the Rock files:
  * The equity file is malformed (unquoted thousands commas), so it is skipped here and
    handled by load_equities.py instead.
  * Some exports start with '#' comment lines, so they are ignored when reading.
  * Rock caps some downloads at 1,000 rows. The inventory flags any file with exactly 1,000 rows.
"""
from pathlib import Path

import pandas as pd

RAW = Path("C:/Users/KIRONJI/Desktop/Rock research project/data/raw")
SKIP = ("equity",)  # files handled by load_equities.py
OUT = Path("C:/Users/KIRONJI/Desktop/Rock research project/outputs")
DAYFIRST = False  # set True if dates look like 31/12/2025 (day first)

DATE_HINTS = ("date", "time", "period", "month", "day")
ID_HINTS = ("ticker", "symbol", "code", "company", "bank", "security", "counter",
            "tenor", "indicator", "category", "debt_type", "pair")


def read_any(path):
    """Read a CSV, trying common encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, comment="#")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}")


def find_date_col(df):
    """Find the date column by name first, then by trying the first columns."""
    named = [c for c in df.columns if any(h in str(c).lower() for h in DATE_HINTS)]
    for c in named + list(df.columns[:3]):
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=DAYFIRST)
        if parsed.notna().mean() > 0.8:
            return c, parsed
    return None, None


def find_id_col(df):
    """Find a ticker / company column if the file stacks several series."""
    for c in df.columns:
        if any(h in str(c).lower() for h in ID_HINTS):
            return c
    return None


def freq_label(dates):
    """Guess frequency from the median gap between distinct dates."""
    d = dates.dropna().drop_duplicates().sort_values()
    if len(d) < 3:
        return "too few dates", None
    gaps = d.diff().dt.days.dropna()
    gap = gaps.median()
    if gap <= 4:
        label = "daily"
    elif gap <= 9:
        label = "weekly"
    elif gap <= 35:
        label = "monthly"
    elif gap <= 100:
        label = "quarterly"
    elif gap <= 200:
        label = "semiannual"
    else:
        label = "annual"
    return label, int(gaps.max())


def inspect(path):
    row = {"file": str(path.relative_to(RAW))}
    if any(k in path.name.lower() for k in SKIP):
        row["error"] = "skipped: use load_equities.py (malformed flow column)"
        return row
    try:
        df = read_any(path)
    except Exception as e:  # keep going even if one file is broken
        row["error"] = str(e)
        return row

    date_col, dates = find_date_col(df)
    id_col = find_id_col(df)
    freq, max_gap = freq_label(dates) if dates is not None else ("no date column", None)

    key = [c for c in (date_col, id_col) if c]
    row.update(
        rows=len(df),
        date_col=date_col,
        first_date=dates.min().date() if dates is not None else None,
        last_date=dates.max().date() if dates is not None else None,
        freq_guess=freq,
        max_gap_days=max_gap,
        id_col=id_col,
        n_ids=df[id_col].nunique() if id_col else None,
        ids_preview=", ".join(map(str, df[id_col].dropna().unique()[:8])) if id_col else None,
        dup_rows=int(df.duplicated(subset=key).sum()) if key else int(df.duplicated().sum()),
        possible_1000_row_cap=len(df) == 1000,
        numeric_cols=int(df.select_dtypes("number").shape[1]),
        missing_pct=round(df.isna().mean().mean() * 100, 2),
        worst_missing_col=df.isna().mean().idxmax(),
        columns=" | ".join(map(str, df.columns)),
    )
    return row


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(RAW.rglob("*.csv"))
    if not files:
        raise SystemExit("No CSV files found in data/raw")

    rows = [inspect(f) for f in files]
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "data_inventory.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", None, "display.max_colwidth", 50)
    show = ["file", "rows", "first_date", "last_date", "freq_guess", "max_gap_days",
            "n_ids", "dup_rows", "missing_pct", "possible_1000_row_cap"]
    print(inv[[c for c in show if c in inv.columns]].to_string(index=False))

    print("\nColumns and tickers per file:")
    for r in rows:
        print(f"- {r['file']}")
        if "error" in r:
            print(f"    ERROR: {r['error']}")
            continue
        print(f"    columns: {r['columns']}")
        if r.get("ids_preview"):
            print(f"    ids ({r['id_col']}): {r['ids_preview']}")

    print(f"\nSaved: {OUT / 'data_inventory.csv'}")


if __name__ == "__main__":
    main()
