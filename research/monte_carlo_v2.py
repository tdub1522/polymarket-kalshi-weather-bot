"""
Monte Carlo v2 — Kalshi Weather NO Strategy.
Uses real Becker trade prices joined to backtest signals.
Filters: T-contracts, GFS NO prediction, distance >= 3.5F, model_prob <= 10%.
"""
try:
    import numpy as np
except ImportError:
    print("Missing dependency: pip install numpy")
    raise SystemExit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Missing dependency: pip install matplotlib")
    raise SystemExit(1)

import polars as pl
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RESEARCH_DIR   = Path(__file__).parent
RESULTS_PATH   = RESEARCH_DIR / "backtest_results.parquet"
TRADES_PATH    = Path("/Users/treywoolley/Desktop/prediction-market-analysis/data/kalshi/trades_consolidated.parquet")
OUTPUT_PATH    = RESEARCH_DIR / "monte_carlo_v2.png"

STARTING_BANKROLL  = 80.0
TRADES_PER_SIM     = 200
POSITION_SIZE      = 3.0
N_SIMULATIONS      = 10_000
RUIN_THRESHOLD     = 20.0
DOUBLE_THRESHOLD   = STARTING_BANKROLL * 2   # $160
SAMPLE_PATHS       = 200

# ── 1. Load & filter backtest signals ─────────────────────────────────────────
df = pl.read_parquet(RESULTS_PATH)

signals = df.filter(
    (pl.col("direction") == "below") &
    (pl.col("signal_side") == "no") &
    ((pl.col("mean_high") - pl.col("threshold_f")).abs() >= 3.5) &
    (pl.col("model_prob") <= 0.10)
)

print(f"Signals after strategy filters: {len(signals)}")

# ── 2. Join real trade prices from Becker ────────────────────────────────────
trades = pl.read_parquet(TRADES_PATH)
trades = trades.filter(pl.col("ticker").str.contains("KXHIGH"))

median_prices = (
    trades
    .group_by("ticker")
    .agg(pl.col("yes_price").median().alias("yes_price_cents"))
)

signals = (
    signals
    .join(median_prices, on="ticker", how="inner")
    .with_columns([
        pl.col("yes_price_cents").cast(pl.Int32),
        ((100 - pl.col("yes_price_cents")) / 100.0).alias("no_cost"),
        (pl.col("yes_price_cents") / (100 - pl.col("yes_price_cents"))).alias("payout_ratio"),
    ])
)

print(f"Signals after price join: {len(signals)}")

# ── 3. Price filter ───────────────────────────────────────────────────────────
signals = signals.filter(
    (pl.col("yes_price_cents") >= 5) &
    (pl.col("yes_price_cents") <= 30) &
    (pl.col("no_cost") >= 0.01)
)

print(f"Signals after price filter (5–30c): {len(signals)}")

n_signals = len(signals)
win_rate  = signals["signal_correct"].mean()

if n_signals < 10:
    print("\nNot enough signals for simulation. Exiting.")
    raise SystemExit(0)

# ── 4. Signal stats ───────────────────────────────────────────────────────────
avg_yes_cents   = signals["yes_price_cents"].mean()
avg_no_cost     = signals["no_cost"].mean()
avg_payout      = signals["payout_ratio"].mean()

print("\n" + "=" * 60)
print("FILTERED SIGNAL STATS")
print("=" * 60)
print(f"  Total signals      : {n_signals:,}")
print(f"  Win rate           : {win_rate:.1%}")
print(f"  Avg YES price      : {avg_yes_cents:.1f}c")
print(f"  Avg NO cost        : {avg_no_cost * 100:.1f}c")
print(f"  Avg payout ratio   : {avg_payout:.2f}x")

print("\n  Win rate by city:")
by_city = (
    signals
    .group_by("city")
    .agg([
        pl.len().alias("count"),
        pl.col("signal_correct").mean().alias("win_rate"),
    ])
    .sort("win_rate", descending=True)
)
print(by_city)

print("\n  Win rate by YES price bucket:")
price_bucketed = signals.with_columns(
    pl.when(pl.col("yes_price_cents") <= 5).then(pl.lit("1-5c"))
    .when(pl.col("yes_price_cents") <= 10).then(pl.lit("6-10c"))
    .when(pl.col("yes_price_cents") <= 15).then(pl.lit("11-15c"))
    .when(pl.col("yes_price_cents") <= 20).then(pl.lit("16-20c"))
    .otherwise(pl.lit("21-30c"))
    .alias("price_bucket")
)
by_price = (
    price_bucketed
    .group_by("price_bucket")
    .agg([
        pl.len().alias("count"),
        pl.col("signal_correct").mean().alias("win_rate"),
    ])
    .sort("price_bucket")
)
print(by_price)

# ── 5. Monte Carlo ────────────────────────────────────────────────────────────
payout_ratios = signals["payout_ratio"].to_numpy()

rng = np.random.default_rng(seed=42)

all_paths      = np.zeros((N_SIMULATIONS, TRADES_PER_SIM + 1))
all_paths[:, 0] = STARTING_BANKROLL

for sim in range(N_SIMULATIONS):
    idx      = rng.integers(0, n_signals, size=TRADES_PER_SIM)
    payouts  = payout_ratios[idx]
    outcomes = rng.random(TRADES_PER_SIM) < win_rate

    bankroll = STARTING_BANKROLL
    for t in range(TRADES_PER_SIM):
        if outcomes[t]:
            bankroll += POSITION_SIZE * payouts[t]
        else:
            bankroll -= POSITION_SIZE
        all_paths[sim, t + 1] = bankroll

final_bankrolls = all_paths[:, -1]

def max_drawdown(path):
    peak = np.maximum.accumulate(path)
    return (peak - path).max()

drawdowns = np.array([max_drawdown(all_paths[i]) for i in range(N_SIMULATIONS)])

# ── 6. Monte Carlo results ────────────────────────────────────────────────────
median_final = np.median(final_bankrolls)
p5_final     = np.percentile(final_bankrolls, 5)
p95_final    = np.percentile(final_bankrolls, 95)
prob_ruin    = np.mean(np.any(all_paths < RUIN_THRESHOLD, axis=1))
prob_double  = np.mean(final_bankrolls >= DOUBLE_THRESHOLD)
median_dd    = np.median(drawdowns)
p95_dd       = np.percentile(drawdowns, 95)

print("\n" + "=" * 60)
print(f"MONTE CARLO RESULTS  ({N_SIMULATIONS:,} sims × {TRADES_PER_SIM} trades, ${POSITION_SIZE:.0f}/trade)")
print("=" * 60)
print(f"  Starting bankroll       : ${STARTING_BANKROLL:.2f}")
print(f"  Median final bankroll   : ${median_final:.2f}")
print(f"  5th pct  final bankroll : ${p5_final:.2f}  (worst case)")
print(f"  95th pct final bankroll : ${p95_final:.2f}  (best case)")
print(f"  Probability of ruin     : {prob_ruin:.1%}  (drops below ${RUIN_THRESHOLD:.0f})")
print(f"  Probability of doubling : {prob_double:.1%}  (reaches ${DOUBLE_THRESHOLD:.0f}+)")
print(f"  Median max drawdown     : ${median_dd:.2f}")
print(f"  95th pct max drawdown   : ${p95_dd:.2f}")

# ── 7. Sharpe ratio ───────────────────────────────────────────────────────────
sharpe_ratios = np.zeros(N_SIMULATIONS)
for sim in range(N_SIMULATIONS):
    idx        = rng.integers(0, n_signals, size=TRADES_PER_SIM)
    payouts_s  = payout_ratios[idx]
    outcomes_s = rng.random(TRADES_PER_SIM) < win_rate
    returns    = np.where(outcomes_s, payouts_s, -1.0)
    std_r      = returns.std()
    sharpe_ratios[sim] = returns.mean() / std_r if std_r > 0 else 0.0

median_sharpe = np.median(sharpe_ratios)
p5_sharpe     = np.percentile(sharpe_ratios, 5)
p95_sharpe    = np.percentile(sharpe_ratios, 95)

print("\n" + "=" * 60)
print("SHARPE RATIO  (per-trade returns, risk-free = 0)")
print("=" * 60)
print(f"  Median Sharpe  : {median_sharpe:.3f}")
print(f"  5th pct Sharpe : {p5_sharpe:.3f}  (worst case)")
print(f"  95th pct Sharpe: {p95_sharpe:.3f}  (best case)")
print(f"  Note: > 1.0 is good, > 2.0 is excellent")

# ── 8. Position size sensitivity ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("POSITION SIZE SENSITIVITY  ($1–$20/trade)")
print("=" * 60)
recommended_size = POSITION_SIZE
crossed_one_pct  = False
for test_size in range(1, 21):
    scale        = test_size / POSITION_SIZE
    scaled_paths = (all_paths - STARTING_BANKROLL) * scale + STARTING_BANKROLL
    ruin_p       = np.mean(np.any(scaled_paths < RUIN_THRESHOLD, axis=1))
    median_br    = np.median(scaled_paths[:, -1])
    if not crossed_one_pct and ruin_p > 0.01:
        crossed_one_pct  = True
        recommended_size = test_size - 1
    marker = " <-- recommended max" if test_size == recommended_size else ""
    print(f"  ${test_size:>2}/trade  →  ruin {ruin_p:.1%}  |  median ${median_br:.0f}{marker}")

if not crossed_one_pct:
    recommended_size = 20
print(f"\n  Recommended max position: ${recommended_size}/trade")

# ── 9. Equity curve chart ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
trade_axis = np.arange(TRADES_PER_SIM + 1)

sample_idx = rng.integers(0, N_SIMULATIONS, size=SAMPLE_PATHS)
for i in sample_idx:
    ax.plot(trade_axis, all_paths[i], color="gray", alpha=0.10, linewidth=0.5)

median_path = np.median(all_paths, axis=0)
p5_path     = np.percentile(all_paths, 5, axis=0)
p95_path    = np.percentile(all_paths, 95, axis=0)

ax.plot(trade_axis, median_path, color="green", linewidth=2.0, label="Median path")
ax.plot(trade_axis, p5_path,    color="red",   linewidth=2.0, label="5th percentile")
ax.plot(trade_axis, p95_path,   color="blue",  linewidth=2.0, label="95th percentile")
ax.axhline(STARTING_BANKROLL, color="steelblue", linewidth=1.2,
           linestyle="--", label=f"Starting bankroll (${STARTING_BANKROLL:.0f})")
ax.axhline(RUIN_THRESHOLD, color="darkred", linewidth=1.0,
           linestyle=":", alpha=0.6, label=f"Ruin threshold (${RUIN_THRESHOLD:.0f})")

ax.set_title(
    f"Monte Carlo: Kalshi Weather NO Strategy (10,000 simulations)\n"
    f"Win rate {win_rate:.1%}  |  ${POSITION_SIZE:.0f}/trade  |  "
    f"{TRADES_PER_SIM} trades  |  Avg payout {avg_payout:.2f}x",
    fontsize=12,
)
ax.set_xlabel("Trade number")
ax.set_ylabel("Bankroll ($)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"\nChart saved to {OUTPUT_PATH}")
