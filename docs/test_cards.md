# Test Cards: Market Research Challenge (Rock Data Plus)

Written: 9 Oct 2026, before any regression was run. Amended 12 Oct 2026 (v1.2), still before any regression was run; every amendment is in the Card change log at the bottom.
Status: FINAL once committed. After the first model run NO card may change.
Hypotheses are frozen (PLAN v2.1). Result fields stay blank until Phase 4.

---

## Common rules (apply to all five cards)

- Outcome series built from the five banks (Absa, KCB, Co-op, Equity, NCBA): returns use the equal-weighted index; flow ratios use the ratio of sums (sector net foreign flow / sector traded value, `flow_ratio`); trading activity uses average daily traded value [change log 1]
- Flow-data integrity rules [change log 7]: a bank-day is excluded from all flow numerators, flow denominators and traded value if (a) volume equals price, (b) the same non-zero volume repeats on 3 or more consecutive trading days, or (c) volume is zero but net flow is not. A weekly or monthly flow ratio or traded-value average needs at least 3 usable days, otherwise it is blank. Prices and returns are not affected. Bank-days where |net flow| exceeds turnover but volume looks plausible are kept in the primary tests and capped in a sensitivity run.
- Standard errors: HAC (Newey-West); max lags 4 for weekly cards, 3 for monthly cards
- Control: NSE 20 return in the SAME period as the outcome (lag 0)
- Sensitivity table (reported on every card, NOT a robustness check): primary test with and without the NSE 20 control. Flow cards (H1, H2, H5) add: equal-weighted mean of the five bank ratios (`flow_ratio_eqw`) and flows capped at +/- turnover (`flow_ratio_capped`). Card-specific rows are listed on each card.
- Week t = Monday-Friday trading week. "Lag 1" means the driver is fully observed before week t starts
- Daily series (interbank rate) become weekly by weekly average; FX by period-end
- Split check: first vs second half at the median period of the panel (unless the card names a date)
- Stationarity: ADF on every series; difference and retest if needed (Phase 3)
- Verdicts:
  - Supported     = expected sign, p < 0.10, and at least 2 of 3 robustness checks pass
  - Opposite sign = reverse sign, p < 0.10, and at least 2 of 3 robustness checks pass
  - Not supported = p >= 0.10 AND the two-sided 90% interval for the standardized effect lies entirely inside (-0.20, +0.20) [change log 15]. Standardized effect = coefficient x SD(driver) / SD(outcome), SDs taken over the panel's full window: the change in the outcome, in outcome standard deviations, for a one-standard-deviation move in the driver
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
Frequency and window: weekly, 17 Feb 2023 - Jun 2026 (174 weeks; the lagged move exceeds 0.2% in 83 of them). FX data genuinely starts 17 Feb 2023 (not an export cap) [change log 6, 10]
Lag: 1 week. Reason: investors react to a move they have already seen.
Expected sign: negative
Primary test: flow ratio(t) on %chg USD/KES(t-1), HAC
Robustness:
  1. Drop one bank at a time
  2. Monthly frequency, same logic (1-month lag). Monthly panel has only 39 rows: the check has low power and the report must say so [change log 11]
  3. Split at 22 Jan 2024 (depreciation episode vs everything after). Verified: USD/KES peak 161.36 on 22 Jan 2024 in Rock's file. Applied by outcome week: part 2 starts with the week ending 26 Jan 2024 (47 vs 127 rows) [change log 3]
Sensitivity rows (not checks): without NSE 20; flow_ratio_eqw; flow_ratio_capped; primary test WITHOUT the two rows with the largest absolute lagged FX change (panel rows 2024-02-23 and 2024-03-22; the FX moves occurred in the weeks ending 16 Feb and 15 Mar 2024) [change log 8]; primary test without the week ending 19 Sep 2025 [change log 13]. This row exists because of statistical leverage, not because the moves are errors: they are real (Eurobond buyback news).
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
Frequency and window: weekly, Sep 2017 - Jun 2026 (457 weeks after the integrity rules blank 3 weeks)
Lag: 1 week. Reason: investors reallocate after yields move; lag 0 mixes in same-week reverse effects.
Expected sign: negative
Primary test: flow ratio(t) on yield change(t-1), HAC
Robustness:
  1. Drop one bank at a time
  2. Swap the driver to the change in the (91-day minus 364-day) yield spread, in points, Monday to Monday (`dspread91m364_lag1`). Sign convention chosen so a short-end-led tightening is positive, like the primary driver; with (364 minus 91) the coefficient would flip sign mechanically. Passes if the sign matches the primary [change log 9]. Two rows are blank because of a source fault in the 364-day series on 25 Aug 2025 [change log 12]
  3. First vs second half (median week)
Sensitivity rows (not checks): without NSE 20; flow_ratio_eqw; flow_ratio_capped. Also a row without the largest sector outflow week (week ending 19 Sep 2025; 17 Sep 2025 carries about 99% of it) [change log 13].
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
Lag: 1 month. Reason: a month's inflation is published after the month ends, so it must be lagged to use only public information.   (Verified: inflation is published on the last day of its reference month, so lag 1 month uses only public information) [change log 4]
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
Frequency and window: weekly, Sep 2017 - Jun 2026 (453 weeks after the integrity rules, confirmed on the real liquidity file) [change log 10]
Lag: 1 week. Reason: funding tightness works through market activity with a delay.
Expected sign: negative
Primary test: avg daily traded value growth(t) on interbank rate change(t-1), HAC
Robustness:
  1. Drop one bank at a time
  2. Monthly frequency, same logic (1-month lag; about 105 months)
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
Frequency and window: monthly, Sep 2017 - Apr 2025 (92 months; the lag costs no rows because debt history starts in 1999) [change log 2]
Lag: 3 months [change log 5]. Reason: debt figures are published with a delay; the Treasury Monthly Debt Bulletins for Dec 2021 and May 2022 sit in upload folders two calendar months after their reference month, so debt for month R is public during month R+2 and only data up to month t-3 is known before month t starts. STATUS (confirmed 12 Oct 2026): the Treasury publishes the debt bulletin at the end of every month, and on 12 Oct the latest bulletin is July 2026 (debt as at the end of July). So the bulletin for month R appears at the end of month R+2, and only data up to month t-3 is public before month t starts. Lag 3 stands.
Expected sign: negative
Primary test: flow ratio(t) on debt growth(t-3), HAC (`dg_total_lag3`)
Robustness:
  1. Drop one bank at a time
  2. Swap the driver to DOMESTIC debt growth only, lag 3 (`dg_dom_lag3`; free of shilling revaluation of external debt)
  3. First vs second half (median month)
Sensitivity table (not checks): lag 2 months (`dg_total_lag2`, the originally planned lag; replaces lag 1); without NSE 20; flow_ratio_eqw; flow_ratio_capped
Limits to state: total debt growth in shillings includes FX revaluation, so H5 partly echoes H1; slow-moving series; no rating or Eurobond signals.
Expected confidence: Low. A wide interval is likely, which means "Inconclusive" rather than "Not supported".
Result: (Phase 4)
```

---

## Card change log (all entries written BEFORE any model was run)

Rule: dated, with a reason. After the first model run no card may change.

| # | Date | Card | Change | Reason |
|---|---|---|---|---|
| 1 | 12 Oct | H1, H2, H5 | Flow outcome is `flow_ratio` = ratio of sums (sector net flow / sector traded value). `flow_ratio_eqw` (mean of five bank ratios) is a sensitivity | "Five-bank index" was ambiguous for flows |
| 2 | 12 Oct | H5 | Sample is 92 rows (lag costs no rows) | Debt history starts 1999; card said "~90" |
| 3 | 12 Oct | H1 | Split date verified (peak 161.36 on 22 Jan 2024). [VERIFY] tag removed | Checked in Rock's FX file |
| 4 | 12 Oct | H3 | Release timing verified (last day of reference month). [VERIFY] tag removed | Checked against KNBS values |
| 5 | 12 Oct | H5 | Primary lag 3 months; check 2 domestic debt lag 3; sensitivity lag 2 (replaces lag 1) | Bulletin for month R-2 appears at the end of month R, so only data to month t-3 is public before month t starts (user approved). Confirmed 12 Oct: latest bulletin on the Treasury page is July 2026, so the bulletin for month R appears at the end of month R+2 |
| 6 | 12 Oct | H1 | No window change | FX data starts 17 Feb 2023 by availability, not an export cap |
| 7 | 12 Oct | H1, H2, H4, H5 | Flow-data integrity rules (see Common rules); capped-flow sensitivity | 36 bank-days (0.3%) had corrupt or stale volume: 30 Apr 2024 (all banks, volume equals price), 26 Jul - 2 Aug 2019 (all banks, stale volume), Absa 28 Nov 2018. Net flow cannot exceed turnover. User approved the patterns |
| 7b | 12 Oct | H1, H2, H5 | Days with a flow but missing volume (for example 31 Aug 2018, 8 Mar 2019) no longer enter the numerator without a denominator | Latent bias in the first build |
| 8 | 12 Oct | H1 | Sensitivity row without the two largest absolute lagged-FX-change rows | Leverage (user approved) |
| 9 | 12 Oct | H2 | Check 2 driver is the change in (91-day minus 364-day) | Avoids a mechanical sign flip (user approved) |
| 10 | 12 Oct | H1, H2, H4 | Text fixes: H1 movement counts; H2 457 rows; H4 453 rows (confirmed) | Counts from the rebuilt panels |
| 11 | 12 Oct | H1, H4 | Monthly panels built for check 2; H1 monthly has 39 rows | Low power must be stated |
| 12 | 12 Oct | H2 | 364-day yield on 25 Aug 2025 set to missing (equals the 182-day value 8.07 between 9.58 and 9.57) | Copy error in the source; affects check 2 only |
| 13 | 12 Oct | H1, H2 | Sensitivity row without the largest sector outflow week (week ending 19 Sep 2025) | One block day (17 Sep 2025) is the biggest flow in the sample. User approved |
| 14 | 12 Oct | H2 | 91-day yield spike on Mon 21 Oct 2019 (6.37 -> 7.25 -> 6.42) is KEPT, no masking. Affects driver rows 1 Nov and 8 Nov 2019 | The event is real: the President refused to sign the Finance Bill 2019 on 16 Oct 2019 and asked MPs to scrap the lending-rate cap. The exact 7.25 print is not independently confirmed |
| 15 | 12 Oct | All | "Tight interval" defined: 90% interval for the standardized effect inside (-0.20, +0.20) (see Common rules) | The card gave no number. User approved |
| 16 | 12 Oct | All | PROPOSED, needs user OK: every sensitivity row in this log is reported; a row may be removed only for a computational defect found independently of its result, and the removal is logged | Prevents dropping rows because they contradict the primary result |

Status of cause-of-event research: the research notes on the top flow weeks are leads, not evidence. No week was dropped because of them, and the report may state a cause only if a retrievable source is cited.
