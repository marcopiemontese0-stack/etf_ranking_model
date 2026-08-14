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

ROLLING_MA = 200      # long moving average window (trading days)
MIN_PERIOD_MA = 100   # minimum periods to compute the moving average
INDICATORS = ["TRADITIONAL", "MOMENTUM", "LONGTERM", "BUYDIP"]
INDICATORS_MIN = 3  # minimum number of valid indicators to keep an ETF
WEIGHTS = {"TRADITIONAL": 0.25, "MOMENTUM": 0.25, "LONGTERM": 0.25, "BUYDIP": 0.25}
MIN_ETFS = 5   # minimum number of valid ETFs




# =============================================================================
# 3. INDICATORS
# =============================================================================

# TRADITIONAL — mean reversion: how far the price sits below its long-term average
long_avg    = prices.rolling(ROLLING_MA, min_periods=MIN_PERIOD_MA).mean()
TRADITIONAL = -(prices - long_avg) / long_avg

# MOMENTUM — trend follower: 12-1 month return (skips the last month, which tends to reverse) + 3-month return
mom_12_1 = prices.shift(21) / prices.shift(260) - 1
mom_3m   = prices / prices.shift(22 * 3) - 1
MOMENTUM = 0.6 * mom_12_1 + 0.4 * mom_3m

# LONGTERM — buy and hold: 2-year Sharpe ratio, penalised by the max drawdown over the same window
ret      = prices.pct_change()
window   = ret.rolling(260 * 2, min_periods=260)
sharpe   = window.mean() / window.std() * np.sqrt(260)
drawdown = prices.rolling(260 * 2, min_periods=260).apply(
    lambda x: (x / np.maximum.accumulate(x) - 1).min(), raw=True
)
LONGTERM = sharpe * (1 + drawdown)   # drawdown is negative -> it penalises

# BUYDIP — buy the dip: rebound from the 3-month low + distance from the 1-year high
min_3m  = prices.rolling(3 * 22).min()
max_1y  = prices.rolling(260).max()
depth   = (prices - min_3m) / min_3m
context = -(prices - max_1y) / max_1y
BUYDIP  = 0.5 * depth + 0.5 * context

# assembly: from 4 DataFrames (date × ticker) to a dict {ticker: DataFrame(date × indicators)}
panel = pd.concat({"TRADITIONAL": TRADITIONAL,
                   "MOMENTUM": MOMENTUM,
                   "LONGTERM": LONGTERM,
                   "BUYDIP": BUYDIP}, axis=1)

names = dict(zip(tickers["ticker"], tickers["name"]))





# =============================================================================
# 4. COMPOSITE SCORE
# =============================================================================
# Stessa logica cross-sectional di prima, ripetuta a OGNI data invece che solo
# sull'ultima. Nessun look-ahead: a ogni giro usa solo dati <= d.

w = pd.Series(WEIGHTS)[INDICATORS]
composite_ts, rank_ts, results = {}, {}, {}

for d in prices.index:

    # 4.1 snapshot: valore di ogni indicatore alla data d, per ogni ETF
    snapshot = panel.loc[d].unstack(level=0)[INDICATORS]   # ticker x indicatori

    # scarta gli ETF con troppi pochi indicatori validi (storia insufficiente)
    included = snapshot.notna().sum(axis=1)
    excluded = snapshot.index[included < INDICATORS_MIN].tolist()
    last = snapshot.loc[included >= INDICATORS_MIN]

    # nelle date di warm-up la cross-section e' troppo piccola: si salta la data
    # (prima era un raise SystemExit, qui deve essere un continue)
    if len(last) < MIN_ETFS:
        continue

    # 4.2 z-score cross-sectional: confronta ogni ETF con gli ALTRI alla stessa data
    raw_z = (last - last.mean()) / last.std()

    # 4.3 ortogonalizzazione simmetrica (Loewdin)
    filled  = raw_z.fillna(0.0)
    corr    = filled.corr().values
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    inv_sqrt_corr = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
    z = pd.DataFrame(filled.values @ inv_sqrt_corr, index=raw_z.index, columns=raw_z.columns)

    # 4.4 composite
    num = z[w.index].mul(w).sum(axis=1)
    den = z[w.index].notna().mul(w).sum(axis=1)
    z["COMPOSITE"] = (num / den).where(den > 0)
    z = z.dropna(subset=["COMPOSITE"])
    z["RANK"] = z["COMPOSITE"].rank(ascending=False).astype(int)

    # 4.5 accumula i risultati della data d
    results[d] = (z, raw_z, excluded)      # snapshot completo, per le diagnostiche
    composite_ts[d] = z["COMPOSITE"]
    rank_ts[d] = z["RANK"]


# 4.6 serie storiche date x ticker: questo e' l'input del backtest
composite_ts = pd.DataFrame(composite_ts).T.reindex(columns=prices.columns)
rank_ts      = pd.DataFrame(rank_ts).T.reindex(columns=prices.columns)
composite_ts.index.name = rank_ts.index.name = "date"

# 4.7 snapshot dell'ultima data, per heatmap e diagnostiche
date = composite_ts.index[-1]
z, raw_z, excluded = results[date]
ranking = z.sort_values("COMPOSITE", ascending=False)
ranking.insert(0, "NAME", [names.get(t, "") for t in ranking.index])
ranking.attrs["excluded"] = excluded
ranking.attrs["raw_z"] = raw_z

print(f"\nAnalysis Date:\n  {date:%d/%m/%Y}")
print(f"Ranking disponibili dal {composite_ts.index[0]:%d/%m/%Y} "
      f"({len(composite_ts)} date su {len(prices)})")





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

# Separator line between the four indicators and the final composite score.
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
# the composite falls below that of the individual signals. Run this on the RAW signals
# (before orthogonalisation): that is where the redundancy shows up, because the
# orthogonalisation removes it by construction.

signals = ranking.attrs["raw_z"]      
# signals = ranking                     # second diagnostic (orthogonalised signals)

correlations = signals[INDICATORS].astype(float).corr()
print("\nCorrelation between signals (cross-section):")
print(correlations.round(2).to_string())

weight_sum = sum(WEIGHTS.values())
composite = signals[INDICATORS].astype(float).mul(pd.Series(WEIGHTS)).sum(axis=1) / weight_sum
composite_std = composite.std()
avg_signal_std = signals[INDICATORS].astype(float).std().mean()
ratio = composite_std / avg_signal_std

print(f"\nComposite dispersion / average signal dispersion: {ratio:.2f}")
if ratio < 0.8:
    print("  Below 1 = aggregation is cancelling information, not adding it up.")

eigenvalues = np.clip(np.linalg.eigvalsh(correlations.values), 0, None)
effective = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
print(f"Effective number of independent signals: {effective:.2f} out of {len(INDICATORS)}")

