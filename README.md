# The Macro Disconnect: Why Foreign Flows in Kenya's Tier 1 Banks Don't Follow Macro Data

A market research project for the Rock Data Plus Market Research Challenge.
We asked whether macro conditions explain foreign investor behaviour in Kenyan bank
stocks, and what that implies for the next 12 months.

**Sector:** Banking (Absa, KCB, Co-op, Equity, NCBA)
**Data:** Rock Data Plus, equities Sep 2017 to Jun 2026

## The short version

- We wrote five hypotheses (currency, T-bill yields, inflation, liquidity, public debt)
  and froze them **before** running any model. Each was tested once.
- **None was supported.** For four of the five (FX, yields, liquidity, inflation), the data
  rule out large macro effects. The fifth (public debt) is inconclusive: the data are too thin to say.
- Foreign flows are lumpy. A handful of single days drive the biggest flow weeks, and no weekly
  or monthly macro driver can explain a single block trade.
- An exploratory event study found no clear MPC-decision effect on flows or returns.
  Finance Bill passage coincided with a large drop in traded value (7 events; post-hoc,
  descriptive, no verdict weight).
- The 12-month outlook is therefore built on **event risk** (August 2027 election,
  Finance Bill 2027, IMF talks), not on coefficients.

## Results

Effects are for a one-standard-deviation move in the driver. "Standardized interval" is the
90% interval in outcome standard deviations; "Not supported" means p ≥ 0.10 **and** the
interval sits inside ±0.20.

| Hypothesis | Driver (lag) | Window, n | Expected → found | Standardized 90% interval | Verdict |
|---|---|---|---|---|---|
| H1 Currency risk | % change USD/KES (1 wk) | Mar 2023 to Jun 2026, 174 wk | negative → positive | -0.04 to +0.12 | Not supported |
| H2 Competition for money | Change in 91-day T-bill yield (1 wk) | Sep 2017 to Jun 2026, 457 wk | negative → positive | -0.13 to +0.13 | Not supported |
| H3 Real returns | Change in headline inflation (1 mo) | Sep 2017 to Jun 2026, 106 mo | negative → negative | -0.19 to +0.04 | Not supported (borderline) |
| H4 Funding conditions | Change in interbank rate (1 wk) | Sep 2017 to Jun 2026, 453 wk | negative → positive | -0.04 to +0.11 | Not supported |
| H5 Sovereign risk | Total public debt growth (3 mo) | Sep 2017 to Apr 2025, 92 mo | negative → negative | -0.32 to +0.05 | Inconclusive |

Full detail: `outputs/phase4/scorecard.csv` and `results_detail.csv`.

## Method

1. **Five test cards** (`docs/test_cards.md`): hypothesis, driver, lag, expected sign, primary
   test and three robustness checks, all written before any regression. `.
2. **Primary test:** OLS with Newey-West (HAC) standard errors, NSE 20 return as control.
3. **Robustness checks** (drop one bank at a time, a swap or frequency check, a split-sample check)
   and **sensitivity rows**.
4. **Flow-data integrity rules:** bank-days with corrupt or stale volume are excluded
   (volume equals price, repeated volume, zero volume with non-zero flow).
5. **Outlook:** event calendar and scenarios built from verified, sourced dates only.
   Weekly coefficients are **not** multiplied into 12-month effects.

Wording note: "helps predict" never means "causes".

## Repo layout

```
docs/test_cards.md        frozen hypotheses and card change log
PLAN.md                   project plan and decision log
scripts/                  panel building, exploration, models, event study, charts
outputs/phase3/           exploration (ADF, top flow weeks)
outputs/phase4/           scorecard, detailed results, Granger tests, run log
outputs/phase5/           current values, event calendar, scenarios, event study, charts
data/                     Rock Data Plus raw files 


## Limits

- Flows are dominated by single-day block trades. The cause of the 17 Sep 2025 block is unknown and not stated.
- Residual autocorrelation remains on four cards. HAC protects the standard errors, not the fit.
- The NSE 20 contains all five banks, so the control partly mirrors the outcome. Dropping it changes no conclusion.
- Equal-weighted flow outcomes flip sign in H1 and H2; the ratio of sums is dominated by large banks and big days.
- The event study is post-hoc and descriptive, with small event counts (7 Finance Bill events).

## Data and sources

Market data: Rock Data Plus. External values (CBK, KNBS, NSE, IEBC, news) are listed with
links and as-of dates in `outputs/phase5/current_values.csv` and `event_calendar.csv`.

## What I learned

HAC errors, stationarity tests (ADF), Granger tests, and why freezing hypotheses before
modelling keeps a null result honest.
