import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. LOAD DATA
# =============================================================================

URL = "https://raw.githubusercontent.com/marcopiemontese0-stack/etf_ranking_model/main/etf_prices.csv"
df = pd.read_csv(URL, header=None)

tickers = df.iloc[0:2, 1:df.shape[1]].transpose()
tickers["ticker"] = tickers.iloc[:, 1]
tickers["name"] = tickers.iloc[:, 0]
tickers = tickers.drop(columns=[tickers.columns[0], tickers.columns[1]])

prices = df.iloc[1:df.shape[0], 0:df.shape[1]]
prices.columns = prices.iloc[0, :]
prices = prices.drop(prices.index[0])
prices.iloc[:, 0] = pd.to_datetime(prices.iloc[:, 0])
prices = prices.set_index(prices.columns[0]).sort_index()
prices = prices.apply(pd.to_numeric, errors="coerce")
prices.index.name = "date"

# =============================================================================
# 2. SETTINGS
# =============================================================================

INDICATORS = ["LONGTERM", "BUYDIP"]
INDICATORS_MIN = 2  # minimum number of valid indicators to keep an ETF (= all of them)
WEIGHTS = {"LONGTERM": 0.5, "BUYDIP": 0.5}
MIN_ETFS = 5   # minimum number of valid ETFs




# =============================================================================
# 3. INDICATORS
# =============================================================================

# LONGTERM — buy and hold: 2-year Sharpe ratio, penalised by the max drawdown over the same window
ret      = prices.pct_change()
window   = ret.rolling(260 * 2, min_periods=260)
sharpe   = window.mean() / window.std() * np.sqrt(260)
drawdown = prices.rolling(260 * 2, min_periods=260).apply(
    lambda x: (x / np.maximum.accumulate(x) - 1).min(), raw=True
)
LONGTERM = sharpe + 2 * drawdown   # 2 = how much weights drawdown vs Sharpe.

# BUYDIP — buy the dip: rebound from the 3-month low + distance from the 1-year high
min_3m  = prices.rolling(3 * 22).min()
max_1y  = prices.rolling(260).max()
depth   = (prices - min_3m) / min_3m
context = -(prices - max_1y) / max_1y
zs = lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
BUYDIP = 0.5 * zs(depth) + 0.5 * zs(context)

# assembly: from 2 DataFrames (date × ticker) to a dict {ticker: DataFrame(date × indicators)}
panel = pd.concat({"LONGTERM": LONGTERM, "BUYDIP": BUYDIP}, axis=1)
names = dict(zip(tickers["ticker"], tickers["name"]))





# =============================================================================
# 4. COMPOSITE SCORE
# =============================================================================
# Same cross-sectional logic as before, but repeated at EVERY date instead of
# only on the last one. No look-ahead: each pass uses data <= d only.

w = pd.Series(WEIGHTS)[INDICATORS]
composite_ts, rank_ts, results = {}, {}, {}

for d in prices.index:

    # snapshot: value of every indicator at date d, for every ETF
    snapshot = panel.loc[d].unstack(level=0)[INDICATORS]   # ticker x indicators

    # drop ETFs with too few valid indicators (insufficient price history)
    included = snapshot.notna().sum(axis=1)
    excluded = snapshot.index[included < INDICATORS_MIN].tolist()
    last = snapshot.loc[included >= INDICATORS_MIN]

    # during warm-up the cross-section is too small to rank against: skip the date
    if len(last) < MIN_ETFS:
        continue

    # cross-sectional z-score: each ETF is measured against the OTHERS on the same date, not against its own history
    raw_z = (last - last.mean()) / last.std()
    z = raw_z.copy()

    # composite: weighted average over the available indicators.
    # num skips NaNs, den sums only the weights actually used, so the weights are renormalised.
    # With INDICATORS_MIN equal to len(INDICATORS) nothing is ever missing here and den is always 1.
    # This is a safeguard, kept for the case where INDICATORS_MIN is later relaxed.
    num = z[w.index].mul(w).sum(axis=1)
    den = z[w.index].notna().mul(w).sum(axis=1)
    z["COMPOSITE"] = (num / den).where(den > 0)
    z = z.dropna(subset=["COMPOSITE"])
    z["RANK"] = z["COMPOSITE"].rank(ascending=False).astype(int)

    # store the results for date d
    results[d] = (z, raw_z, excluded)

    composite_ts[d] = z["COMPOSITE"]
    rank_ts[d] = z["RANK"]


# date x ticker time series: this is what the backtest consumes
composite_ts = pd.DataFrame(composite_ts).T.reindex(columns=prices.columns)
rank_ts      = pd.DataFrame(rank_ts).T.reindex(columns=prices.columns)
composite_ts.index.name = rank_ts.index.name = "date"

# snapshot of the most recent date, for the heatmap and the diagnostics
date = composite_ts.index[-1]
z, raw_z, excluded = results[date]
ranking = z.sort_values("COMPOSITE", ascending=False)
ranking.insert(0, "NAME", [names.get(t, "") for t in ranking.index])
ranking.attrs["excluded"] = excluded
ranking.attrs["raw_z"] = raw_z

print(f"\nAnalysis Date:\n  {date:%d/%m/%Y}")
print(f"Rankings available from {composite_ts.index[0]:%d/%m/%Y} "
      f"({len(composite_ts)} dates out of {len(prices)})")




# =============================================================================
# 5. HEATMAP
# =============================================================================

cols = [*INDICATORS, "COMPOSITE"]
values = ranking[cols].astype(float)

fig, ax = plt.subplots(figsize=(13, max(4, len(ranking) * 0.5 + 1.5)))
image = ax.imshow(values.values, aspect="auto", cmap="RdYlGn")

ax.set_xticks(range(len(cols)))
ax.set_xticklabels([f"{c}\n({WEIGHTS[c]:.0%})" if c in WEIGHTS else f"{c}\n(weighted)" for c in cols],
                   fontsize=10, fontweight="bold")
ax.set_yticks(range(len(ranking)))
ax.set_yticklabels([f"{t}  —  {ranking.loc[t, 'NAME']}" for t in ranking.index], fontsize=9)
ax.axvline(cols.index("COMPOSITE") - 0.5, color="black", linewidth=1, zorder=5)

for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        ax.text(j, i, f"{values.values[i, j]:.2f}",
                ha="center", va="center", fontsize=8, color="black")

fig.colorbar(image, ax=ax, label="Z-score")
ax.set_title(f"ETF quantitative ranking — z-score per indicator  ({date:%d/%m/%Y})",
             fontsize=13, fontweight="bold", pad=15)
fig.tight_layout()
plt.show()





# =============================================================================
# 6. SIGNAL DIAGNOSTICS
# =============================================================================
# Checks whether the four indicators really carry different information.
# Averaging four highly correlated signals does not produce a more robust signal: if two
# of them point in opposite directions, the average cancels them out and the dispersion of
# the composite falls below that of the individual signals. There is no orthogonalisation
# any more, so this diagnostic is load-bearing: whatever redundancy it reports is
# redundancy that ends up in the composite as-is.

signals = ranking.attrs["raw_z"]

correlations = signals[INDICATORS].astype(float).corr()
print("\nCorrelation between signals (cross-section):")
print(correlations.round(2).to_string())

weight_sum = sum(WEIGHTS.values())
composite = signals[INDICATORS].astype(float).mul(pd.Series(WEIGHTS)).sum(axis=1) / weight_sum
composite_std = composite.std()
avg_signal_std = signals[INDICATORS].astype(float).std().mean()
ratio = composite_std / avg_signal_std

eigenvalues = np.clip(np.linalg.eigvalsh(correlations.values), 0, None)
effective = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
print(f"\nEffective number of independent signals: {effective:.2f} out of {len(INDICATORS)}")

wv = pd.Series(WEIGHTS)[INDICATORS].values
expected = np.sqrt(wv @ correlations.values @ wv) / wv.sum()
print(f"Composite dispersion / average signal dispersion: {ratio:.2f} "
      f"(exact value implied by the correlation matrix: {expected:.2f})")
if abs(ratio - expected) > 0.02:
    print("  Mismatch: weights, NaN handling or standardisation are not doing what the code claims.")






# =============================================================================
# 7. BACKTEST — long-only, out-of-sample
# The score at date t uses only data <= t; performance is measured on the NEXT HOLD days (t -> t+HOLD). 
# We go long the top-N ETFs, equal-weighted, and rebalance every HOLD days. 
# Three benchmarks: equal-weight of the whole universe (isolates the ranking's skill), 
# VWCE buy-and-hold (the global equity market), 
# and a 60/40 VWCE/VAGF portfolio rebalanced to fixed weights (a classic balanced allocation).
# =============================================================================

HOLD, TOPN = 22, 4                              # rebalance ~monthly, hold the best 4 of 22
VWCE, VAGF = "VWCEDE", "VAGFDE"                  # equity and bond ETFs used by the benchmarks

fwd   = prices.shift(-HOLD) / prices - 1        # forward return t -> t+HOLD, aligned at t (no look-ahead)
rebal = composite_ts.index[::HOLD]              # non-overlapping rebalance dates

rows, buckets = [], []
for d in rebal:
    s = composite_ts.loc[d].dropna()            # cross-section of scores at d
    r = fwd.loc[d]                              # forward returns of every ETF from d
    s = s[s.index.isin(r.dropna().index)]       # keep only names with a realised forward return
    if len(s) < 3 * TOPN or pd.isna(r.get(VWCE)) or pd.isna(r.get(VAGF)):
        continue
    longs = s.nlargest(TOPN).index              # the model's long book
    rows.append((d, r[longs].mean(),                       # model
                    r.mean(),                              # equal-weight universe
                    r[VWCE],                               # VWCE buy & hold
                    0.6 * r[VWCE] + 0.4 * r[VAGF],         # 60/40 VWCE/VAGF
                    s.corr(r[s.index], method="spearman")))  # rank IC
    order = s.sort_values(ascending=False).index; k = len(order) // 3   # rank terciles
    buckets.append((r[order[:k]].mean(), r[order[k:2*k]].mean(), r[order[-k:]].mean()))

bt = pd.DataFrame(rows, columns=["date", "model", "ew", "vwce", "b6040", "ic"]).set_index("date")
bk = pd.DataFrame(buckets, columns=["top", "mid", "bottom"])

ppy = 260 / HOLD
def stats(x):                                   # annualised return, vol, Sharpe (rf=0), max drawdown
    eq  = (1 + x).cumprod()
    ann = eq.iloc[-1] ** (ppy / len(x)) - 1
    vol = x.std() * np.sqrt(ppy)
    return ann, vol, (ann / vol if vol else np.nan), (eq / eq.cummax() - 1).min()

BENCH = {"ew": "Equal-weight universe", "vwce": "VWCE buy & hold", "b6040": "60/40 VWCE/VAGF"}

print(f"\nLong-only backtest — {len(bt)} rebalances, hold {HOLD}d, top {TOPN} of {prices.shape[1]}\n")
print(f"{'':24}{'annRet':>9}{'annVol':>9}{'Sharpe':>8}{'maxDD':>8}")
for col, lab in [("model", "Model (long top-4)"), *BENCH.items()]:
    a, v, sh, md = stats(bt[col]); print(f"{lab:24}{a:>9.1%}{v:>9.1%}{sh:>8.2f}{md:>8.1%}")

print(f"\n{'active vs benchmark':24}{'active':>9}{'TE':>8}{'IR':>7}{'hit':>6}{'t':>6}")
for col, lab in BENCH.items():
    act = bt["model"] - bt[col]; te = act.std() * np.sqrt(ppy)
    a = (1 + act).prod() ** (ppy / len(act)) - 1
    print(f"{lab:24}{a:>9.1%}{te:>8.1%}{a/te:>7.2f}{(act>0).mean():>6.0%}"
          f"{act.mean()/act.std()*np.sqrt(len(act)):>6.2f}")

print(f"\nRank IC (Spearman vs fwd {HOLD}d): mean {bt['ic'].mean():.3f}, "
      f"t {bt['ic'].mean()/bt['ic'].std()*np.sqrt(len(bt)):.2f}, hit {(bt['ic']>0).mean():.0%}")
print(f"Monotonicity (avg fwd return): top {bk['top'].mean():.2%} | "
      f"mid {bk['mid'].mean():.2%} | bottom {bk['bottom'].mean():.2%}")


# =============================================================================
# 8. EQUITY LINES
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
(1 + bt["model"]).cumprod().plot(ax=ax, label=f"Model — long top-{TOPN}", lw=2.4)
for col, lab in BENCH.items():
    (1 + bt[col]).cumprod().plot(ax=ax, label=lab, lw=2)
ax.axhline(1, color="black", lw=.6); ax.set_ylabel("Growth of 1"); ax.grid(alpha=.3); ax.legend()
ax.set_title(f"Long-only out-of-sample — monthly rebalance, top {TOPN} of {prices.shape[1]} ETFs",
             fontweight="bold")
fig.tight_layout(); plt.show()


