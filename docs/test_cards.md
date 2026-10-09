# Test Cards: Market Research Challenge (Rock Data Plus)

Written: 9 Oct 2026, before any regression was run.
Status: FINAL once committed. Changes after the first model run must be logged at the bottom with a reason.
Hypotheses are frozen (PLAN v2.1). Result fields stay blank until Phase 4.

---

## Common rules (apply to all five cards)

- Outcome series: equal-weighted five-bank index (Absa, KCB, Co-op, Equity, NCBA)
- Standard errors: HAC (Newey-West); max lags 4 for weekly cards, 3 for monthly cards
- Control: NSE 20 return in the SAME period as the outcome (lag 0)
- Sensitivity table (reported on every card, NOT a robustness check): primary test with and without the NSE 20 control
- Week t = Monday-Friday trading week. "Lag 1" means the driver is fully observed before week t starts
- Daily series (interbank rate) become weekly by weekly average; FX by period-end
- Split check: first vs second half at the median period of the panel (unless the card names a date)
- Stationarity: ADF on every series; difference and retest if needed (Phase 3)
- Verdicts:
  - Supported     = expected sign, p < 0.10, and at least 2 of 3 robustness checks pass
  - Opposite sign = reverse sign, p < 0.10, and at least 2 of 3 robustness checks pass
  - Not supported = p >= 0.10 and the confidence interval is tight around zero
  - Inconclusive  = anything else (wide interval, or sign flips across checks)
- What "pass" means for a robustness check (sign relative to the primary result):
  - Drop-one-bank: passes if the sign is unchanged in at least 4 of the 5 runs
  - Swap or frequency check: passes if the sign matches the primary result
  - Split check: passes if the sign matches in both parts
- Effect size is reported in plain words for every card (including a one-standard-deviation move)
- Confidence (High / Medium / Low) is separate from the verdict and depends on sample size and robustness

---

## H1: Currency risk

```
Hypothesis (frozen): months of shilling weakness are followed by weaker net foreign flows into bank stocks.
Rationale: a weaker shilling cuts the dollar value of Kenyan shares, so foreign holders sell or hold back.
Outcome: weekly flow ratio (net foreign flow / traded value), five-bank index
Driver: weekly % change in USD/KES, period-end (positive = weaker shilling)
Controls: NSE 20 weekly return, same week
Frequency and window: weekly, 17 Feb 2023 - Jun 2026 (~175 weeks, real FX movement in ~58)
Lag: 1 week. Reason: investors react to a move they have already seen.
Expected sign: negative
Primary test: flow ratio(t) on %chg USD/KES(t-1), HAC
Robustness:
  1. Drop one bank at a time
  2. Monthly frequency, same logic (1-month lag)
  3. Split at 22 Jan 2024 (depreciation episode vs everything after)   [VERIFY date in FX data]
Direction check (reported separately, does not change the verdict): Granger tests both ways, FX -> flows and flows -> FX, order 2 weeks
Expected confidence: Low (thin movement)
Result: (Phase 4)
```

## H2: Competition for money

```
Hypothesis (frozen): higher T-bill or yield levels are associated with weaker foreign net buying and weaker bank index returns.
Rationale: T-bills compete for the money that would otherwise go into bank stocks.
Outcome: weekly flow ratio, five-bank index
Driver: weekly change in 91-day T-bill yield, percentage points (Monday to Monday)
Controls: NSE 20 weekly return, same week
Frequency and window: weekly, Sep 2017 - Jun 2026 (~459 weeks)
Lag: 1 week. Reason: investors reallocate after yields move; lag 0 mixes in same-week reverse effects.
Expected sign: negative
Primary test: flow ratio(t) on yield change(t-1), HAC
Robustness:
  1. Drop one bank at a time
  2. Swap the driver to the slope (364-day minus 91-day), change in points
  3. First vs second half (median week)
Secondary outcome (descriptive only): weekly bank index return. CBR overlaid on yields (descriptive only).
Direction check: Granger both ways only if time allows
Note for report: tested as rising yields, not high yield levels, to avoid trend-driven relationships.
Result: (Phase 4)
```

## H3: Real returns

```
Hypothesis (frozen): rising inflation is associated with weaker bank index returns.
Rationale: higher inflation erodes real returns and tends to bring tighter policy, which investors price in.
Outcome: monthly price return, five-bank index
Driver: monthly change in headline inflation, percentage points
Controls: NSE 20 monthly return, same month
Frequency and window: monthly, Sep 2017 - Jun 2026 (~106 months)
Lag: 1 month. Reason: a month's inflation is published after the month ends, so it must be lagged to use only public information.   [VERIFY KNBS release day]
Expected sign: negative
Primary test: bank return(t) on headline inflation change(t-1), HAC
Robustness:
  1. Drop one bank at a time
  2. Swap the driver to core inflation (change in points)
  3. First vs second half (median month)
Reference lines (not tested): CBK target, ceiling, floor   [VERIFY values]
Expected confidence: Low to Medium (noisy monthly returns; overlaps with H2 through policy rates)
Result: (Phase 4)
```

## H4: Funding conditions

```
Hypothesis (frozen): tighter liquidity is associated with lower bank trading activity or returns.
Rationale: when money is tight, banks and brokers have less to lend and trade with, so market activity falls.
Outcome: weekly log change in AVERAGE DAILY traded value of the five banks (average per trading day, so short holiday weeks are not distorted)
Driver: weekly change in average interbank rate, percentage points
Controls: NSE 20 weekly return, same week
Frequency and window: weekly, Sep 2017 - Jun 2026 (~459 weeks)
Lag: 1 week. Reason: funding tightness works through market activity with a delay.
Expected sign: negative
Primary test: avg daily traded value growth(t) on interbank rate change(t-1), HAC
Robustness:
  1. Drop one bank at a time
  2. Monthly frequency, same logic (1-month lag)
  3. First vs second half (median week)
Descriptive only: interbank volume (expected sign unclear), weekly bank index return
Limits to state: interbank rate mixes liquidity with policy moves; no market-wide volume control.
Expected confidence: Low to Medium
Result: (Phase 4)
```

## H5: Sovereign risk

```
Hypothesis (frozen): rising public debt is associated with weaker foreign flows.
Rationale: faster debt build-up raises perceived sovereign risk, and foreigners trim Kenyan exposure.
Outcome: monthly flow ratio = total monthly net foreign flow / total monthly traded value (ratio of sums), five-bank index
Driver: monthly % growth in total public debt (external + domestic)
Controls: NSE 20 monthly return, same month
Frequency and window: monthly, Sep 2017 - Apr 2025 (92 months, ~90 after lag)
Lag: 2 months. Reason: debt figures are published with a delay.   [VERIFY release schedule before locking]
Expected sign: negative
Primary test: flow ratio(t) on debt growth(t-2), HAC
Robustness:
  1. Drop one bank at a time
  2. Swap the driver to DOMESTIC debt growth only (free of shilling revaluation of external debt)
  3. First vs second half (median month)
Sensitivity table (not checks): lag 1 month; without NSE 20
Limits to state: total debt growth in shillings includes FX revaluation, so H5 partly echoes H1; slow-moving series; no rating or Eurobond signals.
Expected confidence: Low. A wide interval is likely, which means "Inconclusive" rather than "Not supported".
Result: (Phase 4)
```
