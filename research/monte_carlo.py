"""
Monte Carlo simulation for highest-confidence weather signals.
Filters to NO / below (T-contract) signals with strong GFS separation.
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
RESULTS_PATH = Path(__file__).parent / "backtest_results.parquet"
OUTPUT_PATH  = Path(__file__).parent / "monte_carlo_results.png"

STARTING_BANKROLL  = 80.0
TRADES_PER_SIM     = 100
POSITION_SIZE      = 2.0   # flat dollars per trade
N_SIMULATIONS      = 10_000
RUIN_THRESHOLD     = 20.0
DOUBLE_THRESHOLD   = STARTING_BANKROLL * 2  # $160
SAMPLE_PATHS       = 100

# ── Load & filter ─────────────────────────────────────────────────────────────
df = pl.read_parquet("research/backtest_results.parquet")

df = df.with_columns(
    (1.0 - pl.col("market_prob")).alias("no_cost")
)

filtered = df.filter(
    (pl.col("signal_side") == "no") &
    (pl.col("direction") == "below") &
    (pl.col("edge").abs() >= 0.08) &
    (pl.col("no_cost") >= 0.01) &
    ((pl.col("mean_high") - pl.col("threshold_f")).abs() >= 3.0)
)

print(f"After signal_side==no: {len(df.filter(pl.col('signal_side') == 'no'))}")
print(f"After direction==below: {len(df.filter((pl.col('signal_side') == 'no') & (pl.col('direction') == 'below')))}")
print(f"After edge filter: {len(df.filter((pl.col('signal_side') == 'no') & (pl.col('direction') == 'below') & (pl.col('edge').abs() >= 0.08)))}")
print(f"After no_cost filter: {len(df.filter((pl.col('signal_side') == 'no') & (pl.col('direction') == 'below') & (pl.col('edge').abs() >= 0.08) & ((1.0 - pl.col('market_prob')) >= 0.01)))}")
print(f"After distance filter: {len(filtered)}")

signals = filtered

n_signals      = len(signals)
win_rate       = signals["signal_correct"].mean()
avg_yes_price  = signals["market_prob"].mean()
avg_no_cost    = signals["no_cost"].mean()
median_no_cost = signals["no_cost"].median()
min_no_cost    = signals["no_cost"].min()
max_no_cost    = signals["no_cost"].max()

print("=" * 60)
print("FILTERED SIGNAL STATS")
print("=" * 60)
print(f"  Total signals  : {n_signals}")
print(f"  Win rate       : {win_rate:.1%}")
print(f"  Avg YES price  : {avg_yes_price * 100:.1f}c")
print(f"  Avg NO cost    : {avg_no_cost * 100:.1f}c")
print(f"  Median NO cost : {median_no_cost * 100:.1f}c")
print(f"  NO cost range  : {min_no_cost * 100:.1f}c – {max_no_cost * 100:.1f}c")

if n_signals < 10:
    print("\nNot enough signals for a reliable simulation. Exiting.")
    raise SystemExit(0)

# ── Monte Carlo ───────────────────────────────────────────────────────────────
# Each trade: risk POSITION_SIZE on NO.
# Win profit = POSITION_SIZE * (1 - no_cost) / no_cost
# Loss       = -POSITION_SIZE
no_costs_arr = signals["no_cost"].to_numpy()

rng = np.random.default_rng(seed=42)

all_paths   = np.zeros((N_SIMULATIONS, TRADES_PER_SIM + 1))
all_paths[:, 0] = STARTING_BANKROLL

for sim in range(N_SIMULATIONS):
    idx      = rng.integers(0, n_signals, size=TRADES_PER_SIM)
    no_costs = no_costs_arr[idx]
    outcomes = rng.random(TRADES_PER_SIM) < win_rate

    bankroll = STARTING_BANKROLL
    for t in range(TRADES_PER_SIM):
        no_cost_t = no_costs[t]
        if outcomes[t]:
            bankroll += POSITION_SIZE * (1.0 - no_cost_t) / no_cost_t
        else:
            bankroll -= POSITION_SIZE
        all_paths[sim, t + 1] = bankroll

final_bankrolls = all_paths[:, -1]

# Maximum drawdown per simulation
def max_drawdown(path):
    peak = np.maximum.accumulate(path)
    dd   = peak - path
    return dd.max()

drawdowns = np.array([max_drawdown(all_paths[i]) for i in range(N_SIMULATIONS)])

# ── Summary stats ─────────────────────────────────────────────────────────────
median_final  = np.median(final_bankrolls)
p5_final      = np.percentile(final_bankrolls, 5)
p95_final     = np.percentile(final_bankrolls, 95)
prob_ruin     = np.mean(np.any(all_paths < RUIN_THRESHOLD, axis=1))
prob_double   = np.mean(final_bankrolls >= DOUBLE_THRESHOLD)
median_dd     = np.median(drawdowns)
p95_dd        = np.percentile(drawdowns, 95)

print("\n" + "=" * 60)
print(f"MONTE CARLO RESULTS  ({N_SIMULATIONS:,} simulations × {TRADES_PER_SIM} trades)")
print("=" * 60)
print(f"  Starting bankroll       : ${STARTING_BANKROLL:.2f}")
print(f"  Position size           : ${POSITION_SIZE:.2f} flat per trade")
print(f"  Median final bankroll   : ${median_final:.2f}")
print(f"  5th pct  final bankroll : ${p5_final:.2f}  (worst case)")
print(f"  95th pct final bankroll : ${p95_final:.2f}  (best case)")
print(f"  Probability of ruin     : {prob_ruin:.1%}  (drops below ${RUIN_THRESHOLD:.0f})")
print(f"  Probability of doubling : {prob_double:.1%}  (reaches ${DOUBLE_THRESHOLD:.0f}+)")
print(f"  Median max drawdown     : ${median_dd:.2f}")
print(f"  95th pct max drawdown   : ${p95_dd:.2f}")

# Recommended max size: find largest flat size where ruin prob stays <= 1%
print("\n" + "=" * 60)
print("POSITION SIZE SENSITIVITY  (ruin prob target <= 1%)")
print("=" * 60)
recommended_size = POSITION_SIZE
for test_size in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
    scale   = test_size / POSITION_SIZE
    scaled  = (all_paths - STARTING_BANKROLL) * scale + STARTING_BANKROLL
    ruin_p  = np.mean(np.any(scaled < RUIN_THRESHOLD, axis=1))
    flag    = " <-- recommended max" if ruin_p <= 0.01 else ""
    if ruin_p <= 0.01:
        recommended_size = test_size
    print(f"  ${test_size:>2}/trade → ruin prob {ruin_p:.1%}{flag}")

print(f"\n  Recommended max position: ${recommended_size}/trade")

# ── Sharpe ratio ──────────────────────────────────────────────────────────────
# Per-trade return: (payout - cost) / cost for wins, -1.0 for losses
# cost = no_cost = 1 - market_prob (price paid per $1 of NO)
RISK_FREE_RATE = 0.0

sharpe_ratios = np.zeros(N_SIMULATIONS)
for sim in range(N_SIMULATIONS):
    idx        = rng.integers(0, n_signals, size=TRADES_PER_SIM)
    no_costs_s = no_costs_arr[idx]
    outcomes_s = rng.random(TRADES_PER_SIM) < win_rate

    returns = np.where(
        outcomes_s,
        (1.0 - no_costs_s) / no_costs_s,   # net profit per dollar risked on a win
        -1.0,                                # full loss
    )
    mean_r = returns.mean()
    std_r  = returns.std()
    sharpe_ratios[sim] = (mean_r - RISK_FREE_RATE) / std_r if std_r > 0 else 0.0

median_sharpe = np.median(sharpe_ratios)
p5_sharpe     = np.percentile(sharpe_ratios, 5)
p95_sharpe    = np.percentile(sharpe_ratios, 95)

print("\n" + "=" * 60)
print("SHARPE RATIO  (per-trade returns, risk-free rate = 0)")
print("=" * 60)
print(f"  Median Sharpe  : {median_sharpe:.3f}")
print(f"  5th pct Sharpe : {p5_sharpe:.3f}  (worst case)")
print(f"  95th pct Sharpe: {p95_sharpe:.3f}  (best case)")
print(f"  Note: Sharpe > 1.0 is good, > 2.0 is excellent for a trading strategy")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
trade_axis = np.arange(TRADES_PER_SIM + 1)

# Sample paths
sample_idx = rng.integers(0, N_SIMULATIONS, size=SAMPLE_PATHS)
for i in sample_idx:
    ax.plot(trade_axis, all_paths[i], color="gray", alpha=0.15, linewidth=0.6)

# Percentile paths
median_path = np.median(all_paths, axis=0)
p5_path     = np.percentile(all_paths, 5, axis=0)

ax.plot(trade_axis, median_path, color="green",  linewidth=2.0, label="Median path")
ax.plot(trade_axis, p5_path,     color="red",    linewidth=2.0, label="5th percentile")
ax.axhline(STARTING_BANKROLL, color="steelblue", linewidth=1.2,
           linestyle="--", label=f"Starting bankroll (${STARTING_BANKROLL:.0f})")
ax.axhline(RUIN_THRESHOLD, color="darkred", linewidth=1.0,
           linestyle=":", alpha=0.7, label=f"Ruin threshold (${RUIN_THRESHOLD:.0f})")

ax.set_title(
    f"Monte Carlo — NO/Below T-contracts  |  {N_SIMULATIONS:,} sims × {TRADES_PER_SIM} trades  |  "
    f"${POSITION_SIZE:.0f}/trade  |  Win rate {win_rate:.1%}",
    fontsize=11,
)
ax.set_xlabel("Trade number")
ax.set_ylabel("Bankroll ($)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"\nEquity curve saved to {OUTPUT_PATH}")

# ── Sharpe distribution chart ──────────────────────────────────────────────────
SHARPE_PATH = Path(__file__).parent / "sharpe_distribution.png"

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.hist(sharpe_ratios, bins=80, color="steelblue", edgecolor="white", linewidth=0.4)
ax2.axvline(median_sharpe, color="green", linewidth=2.0,
            label=f"Median: {median_sharpe:.3f}")
ax2.axvline(1.0, color="orange", linewidth=1.5, linestyle="--", label="Sharpe = 1.0 (good)")
ax2.axvline(2.0, color="red",    linewidth=1.5, linestyle="--", label="Sharpe = 2.0 (excellent)")

ax2.set_title(
    f"Sharpe Ratio Distribution — {N_SIMULATIONS:,} simulations × {TRADES_PER_SIM} trades  |  "
    f"Win rate {win_rate:.1%}",
    fontsize=11,
)
ax2.set_xlabel("Sharpe ratio (per-trade returns)")
ax2.set_ylabel("Simulation count")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SHARPE_PATH, dpi=150)
print(f"Sharpe distribution saved to {SHARPE_PATH}")
