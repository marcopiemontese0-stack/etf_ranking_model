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

# =============================================================================
# 3. INDICATORS
# =============================================================================

def compute_indicators(price: pd.Series) -> pd.DataFrame:
    """Compute the four indicators for a single ETF's price series."""
    d = pd.DataFrame({"price": price})

    # TRADITIONAL - mean reversion: how far the price sits below its long-term average.
    long_avg = d["price"].rolling(ROLLING_MA, min_periods=MIN_PERIOD_MA).mean()
    d["TRADITIONAL"] = -(d["price"] - long_avg) / long_avg

    # MOMENTUM - trend follower: 12-1 month return (skips the last month, which tends to reverse) blended with the 3-month return.
    mom_12_1 = d["price"].shift(21) / d["price"].shift(260) - 1
    mom_3m = d["price"] / d["price"].shift(22*3) - 1
    d["MOMENTUM"] = 0.6 * mom_12_1 + 0.4 * mom_3m

    # LONGTERM - buy and hold: 2-year Sharpe ratio, penalized by the max drawdown over the same window.
    ret = d["price"].pct_change()
    window = ret.rolling(260*2, min_periods=260)
    sharpe = window.mean() / window.std() * np.sqrt(260)
    drawdown = d["price"].rolling(2+260, min_periods=260).apply(
        lambda x: (x / np.maximum.accumulate(x) - 1).min(), raw=True
    )
    d["LONGTERM"] = sharpe * (1 + drawdown)  # drawdown is negative -> penalizes

    # BUYDIP - buy the dip: rebound from the 3-month low plus distance from the 1-year high. 
    min_3m = d["price"].rolling(3*22).min()
    max_1y = d["price"].rolling(260).max()
    depth = (d["price"] - min_3m) / min_3m
    context = -(d["price"] - max_1y) / max_1y
    d["BUYDIP"] = 0.5 * depth + 0.5 * context

    return d[INDICATORS]


indicators = {ticker: compute_indicators(prices[ticker]) for ticker in prices.columns}
names = dict(zip(tickers["ticker"], tickers["name"]))

# =============================================================================
# 4. COMPOSITE SCORE
# =============================================================================

def orthogonalize(z: pd.DataFrame) -> pd.DataFrame:
    """Symmetric (Loewdin) orthogonalization: decorrelates the signals while
    keeping each one as close as possible to its original version, so no
    signal is arbitrarily favored the way sequential (Gram-Schmidt)
    orthogonalization would favor whichever signal goes first.
    """
    filled = z.fillna(0.0)  # 0 = cross-sectional average, a neutral stand-in for missing data
    corr = filled.corr().values
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)  # avoid divide-by-zero on near-collinear signals
    inv_sqrt_corr = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
    orth = filled.values @ inv_sqrt_corr
    return pd.DataFrame(orth, index=z.index, columns=z.columns)


def compute_ranking(indicators: dict, names: dict) -> pd.DataFrame:
    """Cross-sectional z-score at the latest date, orthogonalized and combined
    into a weighted average.

    The z-score compares each ETF with the OTHER ETFs at the same point in time,
    not with its own history: that's what makes it possible to compare indicators
    with different units (a ratio vs. a Sharpe ratio). Orthogonalization then
    removes the overlap between signals so the weighted average doesn't cancel
    itself out (see section 6, signal diagnostics).
    """
    last = pd.DataFrame({t: ind.iloc[-1] for t, ind in indicators.items()}).T[INDICATORS]

    included = last.notna().sum(axis=1)
    excluded = last.index[included < INDICATORS_MIN].tolist()
    last = last.loc[included >= INDICATORS_MIN]

    if len(last) < 5:
        raise SystemExit(
            f"Only {len(last)} ETFs have enough history: with so few names the cross-sectional z-score is not meaningful."
        )

    raw_z = (last - last.mean()) / last.std()
    z = orthogonalize(raw_z)

    def composite(riga):
        membs = [c for c in WEIGHTS if pd.notna(riga[c])]
        if not membs:
            return np.nan
        return sum(WEIGHTS[c] * riga[c] for c in membs) / sum(WEIGHTS[c] for c in membs)

    z["COMPOSITE"] = z.apply(composite, axis=1)
    z = z.dropna(subset=["COMPOSITE"])
    z["RANK"] = z["COMPOSITE"].rank(ascending=False).astype(int)

    ranking = z.sort_values("COMPOSITE", ascending=False)
    ranking.insert(0, "NAME", [names.get(t, "") for t in ranking.index])
    ranking.attrs["excluded"] = excluded
    ranking.attrs["raw_z"] = raw_z  # pre-orthogonalization signals, for diagnostics
    return ranking


ranking = compute_ranking(indicators, names)
ranking

# =============================================================================
# 5. HEATMAP
# =============================================================================

def heatmap(ranking: pd.DataFrame, date: pd.Timestamp):
    cols = [*INDICATORS, "COMPOSITE"]
    values = ranking[cols].astype(float)

    fig, ax = plt.subplots(figsize=(13, max(4, len(ranking) * 0.5 + 1.5)))
    image = ax.imshow(values.values, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        [f"{c}\n({WEIGHTS[c]:.0%})" if c in WEIGHTS else f"{c}\n(weighted)" for c in cols],
        fontsize=10, fontweight="bold",
    )
    ax.set_yticks(range(len(ranking)))
    ax.set_yticklabels(
        [f"{t}  —  {ranking.loc[t, 'NAME']}" for t in ranking.index], fontsize=9
    )

    # Separator line between the four indicators and the final composite score.
    ax.axvline(cols.index("COMPOSITE") - 0.5, color="black", linewidth=3, zorder=5)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(image, ax=ax, label="Z-score")
    ax.set_title(f"ETF quantitative ranking — z-score per indicator  ({date:%d/%m/%Y})",
                 fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    plt.show()
    return fig


_ = heatmap(ranking, prices.index[-1])

# =============================================================================
# 6. SIGNAL DIAGNOSTICS
# =============================================================================

def diagnostics(signals: pd.DataFrame) -> None:
    """Check whether the four indicators actually carry different information.

    Averaging four strongly correlated signals doesn't produce a more robust
    signal: if two of them are opposite to each other, the average cancels
    them out and the composite's dispersion collapses below that of the
    individual signals. Run this on the RAW (pre-orthogonalization) signals:
    that's where redundancy actually shows up, since orthogonalize() removes
    it by construction.
    """
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


diagnostics(ranking.attrs["raw_z"])
diagnostics(ranking)
