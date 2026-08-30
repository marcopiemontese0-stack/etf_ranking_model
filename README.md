# ETF ranking model

A daily cross-sectional ranking of European UCITS ETFs, built on two uncorrelated factors.
The model answers one question: “Among these ETFs, which ones look best today?” It does not answer whether you should be invested at all.

---
## Quick start
```bash
pip install pandas numpy matplotlib
python etf_ranking.py
```
The script reads `etf_prices.csv` directly from this repo, so no local data setup is needed.
Daily prices run from January 2021 to August 2026.

---
## The two factors
**LONGTERM — quality of the ride**
```
LONGTERM = Sharpe(2y) + 2 * MaxDrawdown(2y)
```
Sharpe is annualized over a two-year rolling window. The drawdown is negative, so it always
lowers the score. The multiplier 2 sets how much a drawdown costs relative to a point of
Sharpe. The factor rewards ETFs that went up steadily, and punishes the ones that went up
through deep falls.

**BUYDIP — shape of the fall**
```
depth   = rebound from the 3-month low
context = distance below the 1-year high
BUYDIP  = 0.5 * z(depth) + 0.5 * z(context)
```
Both legs are standardized before they are added, otherwise the leg with the wider spread
would dominate and the 50/50 weights would be fiction. The factor is high only when both
conditions hold (well below the highs but already recovering). It is low for an ETF sitting
at its high, and equally low for one that is still falling.

---
## How the score is built
For every date, the script does the following:
1. Take the value of both factors for every ETF on that date.
2. Drop ETFs without enough price history and skip the date if fewer than 5 remain.
3. Standardize each factor **across ETFs** (not across time), so both have mean 0 and
   standard deviation 1 on that date.
4. Average them 50/50 into a `COMPOSITE` score, then rank. Rank 1 is the best.
The result is two `date x ticker` tables: `composite_ts` and `rank_ts`. Rankings are available
from January 2022, once the two-year windows have enough data. Every rolling window is trailing, so the score at date *d* only uses data up to *d*.

---
## Output
- A heatmap of the latest ranking, with the z-score of each factor and the composite.
- Signal diagnostics printed on the console.
Latest run:
```
Effective number of independent signals: 1.97 out of 2
Correlation between LONGTERM and BUYDIP: 0.12
Composite dispersion / average signal dispersion: 0.75 (exact value implied by C: 0.75)
```
The last line is a self-check, not a statistic. The dispersion of a weighted average of standardized signals must equal `sqrt(w' C w) / sum(w)`. 
If the computed value does not match the algebra, then the weights, the NaN handling or the standardization are not doing what the code claims.


---
## Out-of-sample test

A ranking is only useful if a higher score today leads to a higher return tomorrow.
This section tests exactly that, out of sample: the score at date *d* uses only data up to *d*, and it is judged on the return of the **next** month.

Setup: on each rebalance date, go long the top 4 ETFs, equal-weighted, and hold for one month (22 trading days). 
Rebalance and repeat. 56 non-overlapping months, from January 2022 to August 2026.

Three benchmarks, each answering a different question:
- **Equal-weight universe** — the same 22 ETFs held equally, no selection. The honest one: any outperformance here comes from the ranking, not from the universe.
- **VWCE buy & hold** — the global equity market (FTSE All-World). Answers "could I have just bought the market?".
- **60/40 VWCE/VAGF** — a classic balanced portfolio, rebalanced monthly. A risk reference, not a fair skill test (see below).
```
                           annRet   annVol  Sharpe   maxDD
Model (long top-4)          16.2%    16.3%    0.99  -16.5%
Equal-weight universe       10.8%    12.2%    0.89  -10.4%
VWCE buy & hold             10.1%    13.8%    0.74  -17.5%
60/40 VWCE/VAGF              5.4%     8.9%    0.61  -10.4%

active vs benchmark        active      TE     IR   hit     t
Equal-weight universe        5.2%    8.8%   0.59   59%  1.32
VWCE buy & hold              5.2%   12.7%   0.40   50%  0.98
60/40 VWCE/VAGF             10.5%   12.5%   0.84   63%  1.84

```
Two things matter more than the equity curve: where the edge sits, and how strong it really is.

**The edge is in the top, not in the ordering.** Split the universe into rank thirds
and average the next-month return of each group:
```
Rank IC (Spearman vs fwd 22d): mean 0.039, t 0.98, hit 65%
Monotonicity (avg fwd return): top 1.19% | mid 0.84% | bottom 0.84%
```
The top third clearly beats the field, but the middle and bottom are indistinguishable.
The score identifies the leaders, it does not rank the laggards. 
This is consistent with the long-only design: the value sits in the top of the book, which is also why shorting the bottom names adds nothing.

**Predictive power, measured honestly.** The rank information coefficient (Spearman correlation between score and next-month return) averages +0.039, with the right sign in 65% of months.
But over the ~54 months in the sample its t-stat is about 1.0, well below the usual bar of 2: the average is not statistically distinguishable from zero here. 
Read it as a signal that points the right way more often than not, whose size this short history cannot yet prove.

How to read the three benchmarks together. 
The equal-weight comparison is the one that isolates skill, and it gives an information ratio of [IR vs EW]. 
The 60/40 comparison looks the strongest (0.84) but it is the least meaningful: a fully invested equity book beats a bond-diluted portfolio mostly by collecting the equity risk premium, which is not skill. 
The model also clears the 60/40 on Sharpe (0.99 vs 0.61), so it is not only taking more risk — but the raw return gap should not be read as alpha.



---
## Limitations
Please read this section before using any number from this repo.

**The test is walk-forward, not a full backtest.**
No transaction costs, no bid-ask spread, no slippage. Monthly rebalancing of a 4-ETF book is cheap but not free, and the reported returns are gross.

**The score is relative, never absolute.** 
Because the standardization happens across ETFs on each date, someone always scores +2, even in a market where every ETF is falling. 
The model tells you what to prefer, never whether to buy.

**Survivorship bias.** 
The price file only contains ETFs that exist today. Any historical study on this universe is optimistic by construction, because the products that closed are missing.
This now flatters the out-of-sample test as well, not just a hypothetical backtest.

**New ETFs cannot appear.** 
LONGTERM needs about one year of returns inside a two-year window.
Recently launched ETFs are excluded until they build that history. Two of the 22 are excluded
in the latest run for this reason.

**Sharpe without a risk-free rate.** 
The Sharpe here is a return-to-volatility ratio. 
Since a constant is the same for every ETF, it disappears in the cross-sectional standardization, so the ranking is not affected, but the number should not be read as a true Sharpe ratio.

**Small universe.** 
22 ETFs is a thin cross-section. Standardizing over so few observations makes the mean and the standard deviation noisy, and a single outlier moves everyone else.

**Short sample.** 
The out-of-sample test covers 56 monthly rebalances over about four and a half years. 
That is enough to see a direction, not enough to prove an edge with confidence. The t-stats above say so.

**Two benchmarks live inside the universe.** 
VWCE and VAGF are among the 22 ranked ETFs, so the model sometimes holds them. 
The comparisons against them are therefore not fully independent.

---

## Configuration
All settings are in section 2 of the script.
| Setting | Meaning |
| --- | --- |
| `INDICATORS` | Which factors to use |
| `WEIGHTS` | Weight of each factor in the composite |
| `INDICATORS_MIN` | Minimum valid factors to keep an ETF |
| `MIN_ETFS` | Minimum cross-section size to rank a date |

Weights are renormalized over the available factors, so the model still works if `INDICATORS_MIN` is later relaxed to allow a missing factor.

---
## Design history
The first version used four factors: TRADITIONAL, MOMENTUM, LONGTERM and BUYDIP.

The diagnostics showed the problem. TRADITIONAL and MOMENTUM had a cross-sectional correlation of -0.93, because both are essentially the trailing return with opposite signs. 
The four factors carried only 1.43 independent signals, and the composite had one third of the dispersion of its own inputs. The aggregation was cancelling information, not adding it up.

Dropping to two genuinely different factors was the better answer. Independent signals went from 1.43/4 to 1.97/2, and daily rank turnover fell by roughly 40%.

---
## Disclaimer
This is a personal project for research and learning. It is not investment advice.

