"""
Bracket contract analysis: compare T-contracts (validated) vs B-contracts (new).
Determines whether bracket markets share the same edge as T-contracts.
"""
try:
    import numpy as np
except ImportError:
    print("Missing dependency: pip install numpy")
    raise SystemExit(1)

import polars as pl
from pathlib import Path

RESEARCH_DIR  = Path(__file__).parent
RESULTS_PATH  = RESEARCH_DIR / "backtest_results.parquet"
TRADES_PATH   = Path("/Users/treywoolley/Desktop/prediction-market-analysis/data/kalshi/trades_consolidated.parquet")

N_SIMULATIONS   = 10_000
TRADES_PER_SIM  = 100
STARTING_BANKROLL = 80.0
RUIN_THRESHOLD  = 20.0

# ── Load ──────────────────────────────────────────────────────────────────────
df = pl.read_parquet(RESULTS_PATH)
print(f"Loaded {len(df)} total backtest records\n")

# ── Build trade price lookup ───────────────────────────────────────────────────
trades = pl.read_parquet(TRADES_PATH)
trades = trades.filter(pl.col("ticker").str.contains("KXHIGH"))
median_prices = (
    trades
    .group_by("ticker")
    .agg(pl.col("yes_price").median().alias("yes_price_cents"))
)

# ── Category definitions ───────────────────────────────────────────────────────
# Cat A: T-contracts (direction=below, GFS predicts NO, GFS above threshold)
cat_a = (
    df.filter(
        (pl.col("direction") == "below") &
        (pl.col("signal_side") == "no") &
        (pl.col("mean_high") > pl.col("threshold_f"))
    )
    .with_columns(
        (pl.col("mean_high") - pl.col("threshold_f")).alias("distance")
    )
)

# Cat B: Bracket contracts (direction=above, GFS above TOP of bracket range)
cat_b = (
    df.filter(
        (pl.col("direction") == "above") &
        (pl.col("signal_side") == "no") &
        (pl.col("mean_high") > pl.col("threshold_f") + 0.5)
    )
    .with_columns(
        (pl.col("mean_high") - (pl.col("threshold_f") + 0.5)).alias("distance")
    )
)

DIST_BUCKETS = [
    ("0-1F",  pl.col("distance") < 1),
    ("1-2F",  (pl.col("distance") >= 1) & (pl.col("distance") < 2)),
    ("2-3F",  (pl.col("distance") >= 2) & (pl.col("distance") < 3)),
    ("3-5F",  (pl.col("distance") >= 3) & (pl.col("distance") < 5)),
    ("5F+",   pl.col("distance") >= 5),
]


def dist_bucket_col(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(
        pl.when(pl.col("distance") < 1).then(pl.lit("0-1F"))
        .when(pl.col("distance") < 2).then(pl.lit("1-2F"))
        .when(pl.col("distance") < 3).then(pl.lit("2-3F"))
        .when(pl.col("distance") < 5).then(pl.lit("3-5F"))
        .otherwise(pl.lit("5F+"))
        .alias("dist_bucket")
    )


def prob_bucket_col(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(
        pl.when(pl.col("model_prob") <= 0.05).then(pl.lit("0-5%"))
        .when(pl.col("model_prob") <= 0.10).then(pl.lit("5-10%"))
        .otherwise(pl.lit("10-20%"))
        .alias("prob_bucket")
    )


def price_bucket_col(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(
        pl.when(pl.col("yes_price_cents") <= 5).then(pl.lit("1-5c"))
        .when(pl.col("yes_price_cents") <= 10).then(pl.lit("6-10c"))
        .when(pl.col("yes_price_cents") <= 15).then(pl.lit("11-15c"))
        .when(pl.col("yes_price_cents") <= 20).then(pl.lit("16-20c"))
        .otherwise(pl.lit("21-30c"))
        .alias("price_bucket")
    )


def print_category_stats(label: str, data: pl.DataFrame):
    print("=" * 60)
    print(f"CATEGORY {label}")
    print("=" * 60)
    print(f"  Total contracts : {len(data):,}")
    print(f"  Overall win rate: {data['signal_correct'].mean():.1%}")

    print("\n  Win rate by distance:")
    bucketed = dist_bucket_col(data)
    print(
        bucketed.group_by("dist_bucket")
        .agg([pl.len().alias("count"), pl.col("signal_correct").mean().alias("win_rate")])
        .sort("dist_bucket")
    )

    print("\n  Win rate by city:")
    print(
        data.group_by("city")
        .agg([pl.len().alias("count"), pl.col("signal_correct").mean().alias("win_rate")])
        .sort("win_rate", descending=True)
    )

    print("\n  Win rate by model_prob:")
    print(
        prob_bucket_col(data)
        .group_by("prob_bucket")
        .agg([pl.len().alias("count"), pl.col("signal_correct").mean().alias("win_rate")])
        .sort("prob_bucket")
    )


def apply_price_filter(data: pl.DataFrame) -> pl.DataFrame:
    return (
        data
        .join(median_prices, on="ticker", how="inner")
        .filter(
            (pl.col("yes_price_cents") >= 5) &
            (pl.col("yes_price_cents") <= 30)
        )
        .with_columns(
            (pl.col("yes_price_cents") / (100 - pl.col("yes_price_cents"))).alias("payout_ratio")
        )
    )


def print_price_stats(label: str, data: pl.DataFrame):
    print(f"\n  [{label}] After price filter (5–30c):")
    print(f"    Contracts remaining : {len(data):,}")
    print(f"    Win rate            : {data['signal_correct'].mean():.1%}")
    print(f"    Avg YES price       : {data['yes_price_cents'].mean():.1f}c")
    print(f"\n    Win rate by YES price bucket:")
    print(
        price_bucket_col(data)
        .group_by("price_bucket")
        .agg([pl.len().alias("count"), pl.col("signal_correct").mean().alias("win_rate")])
        .sort("price_bucket")
    )


def run_monte_carlo(data: pl.DataFrame, position_size: float, label: str) -> dict:
    n = len(data)
    win_rate = data["signal_correct"].mean()
    payout_ratios = data["payout_ratio"].to_numpy()

    rng = np.random.default_rng(seed=42)
    all_paths = np.zeros((N_SIMULATIONS, TRADES_PER_SIM + 1))
    all_paths[:, 0] = STARTING_BANKROLL

    for sim in range(N_SIMULATIONS):
        idx      = rng.integers(0, n, size=TRADES_PER_SIM)
        payouts  = payout_ratios[idx]
        outcomes = rng.random(TRADES_PER_SIM) < win_rate
        bankroll = STARTING_BANKROLL
        for t in range(TRADES_PER_SIM):
            bankroll += position_size * payouts[t] if outcomes[t] else -position_size
            all_paths[sim, t + 1] = bankroll

    final = all_paths[:, -1]
    prob_ruin = float(np.mean(np.any(all_paths < RUIN_THRESHOLD, axis=1)))

    # Sharpe
    sharpes = np.zeros(N_SIMULATIONS)
    for sim in range(N_SIMULATIONS):
        idx      = rng.integers(0, n, size=TRADES_PER_SIM)
        payouts  = payout_ratios[idx]
        outcomes = rng.random(TRADES_PER_SIM) < win_rate
        returns  = np.where(outcomes, payouts, -1.0)
        std_r    = returns.std()
        sharpes[sim] = returns.mean() / std_r if std_r > 0 else 0.0

    result = {
        "label":          label,
        "n_signals":      n,
        "win_rate":       win_rate,
        "position_size":  position_size,
        "median_final":   float(np.median(final)),
        "p5_final":       float(np.percentile(final, 5)),
        "p95_final":      float(np.percentile(final, 95)),
        "prob_ruin":      prob_ruin,
        "median_sharpe":  float(np.median(sharpes)),
    }

    print(f"\n  [{label}] Monte Carlo ({N_SIMULATIONS:,} sims × {TRADES_PER_SIM} trades, ${position_size:.0f}/trade):")
    print(f"    Median final bankroll : ${result['median_final']:.2f}")
    print(f"    5th pct final bankroll: ${result['p5_final']:.2f}")
    print(f"    95th pct final bankroll: ${result['p95_final']:.2f}")
    print(f"    Probability of ruin   : {prob_ruin:.1%}")
    print(f"    Median Sharpe ratio   : {result['median_sharpe']:.3f}")

    return result


# ── Per-category analysis ─────────────────────────────────────────────────────
print_category_stats("A — T-contracts (direction=below)", cat_a)
print()
print_category_stats("B — Bracket contracts (direction=above)", cat_b)

# ── Price filter ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRICE FILTER (joined with Becker trades, 5–30c YES)")
print("=" * 60)

cat_a_priced = apply_price_filter(cat_a)
cat_b_priced = apply_price_filter(cat_b)

print_price_stats("A — T-contracts", cat_a_priced)
print_price_stats("B — Bracket contracts", cat_b_priced)

# ── Monte Carlo ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MONTE CARLO SIMULATIONS")
print("=" * 60)

results_a = run_monte_carlo(cat_a_priced, position_size=3.0, label="A — T-contracts")
results_b = run_monte_carlo(cat_b_priced, position_size=2.0, label="B — Bracket contracts")

# ── Side-by-side recommendation table ────────────────────────────────────────
print("\n" + "=" * 60)
print("RECOMMENDATION TABLE")
print("=" * 60)
print(f"{'Metric':<30} {'Cat A (T-contracts)':>20} {'Cat B (Brackets)':>18}")
print("-" * 70)
print(f"{'Signals (pre-price)':.<30} {len(cat_a):>20,} {len(cat_b):>18,}")
print(f"{'Win rate (pre-price)':.<30} {cat_a['signal_correct'].mean():>19.1%} {cat_b['signal_correct'].mean():>17.1%}")
print(f"{'Signals (5-30c)':.<30} {results_a['n_signals']:>20,} {results_b['n_signals']:>18,}")
print(f"{'Win rate (5-30c)':.<30} {results_a['win_rate']:>19.1%} {results_b['win_rate']:>17.1%}")
print(f"{'Position size':.<30} {'$' + str(int(results_a['position_size'])):>20} {'$' + str(int(results_b['position_size'])):>18}")
print(f"{'Median final bankroll':.<30} ${results_a['median_final']:>19.2f} ${results_b['median_final']:>17.2f}")
print(f"{'5th pct final bankroll':.<30} ${results_a['p5_final']:>19.2f} ${results_b['p5_final']:>17.2f}")
print(f"{'Ruin probability':.<30} {results_a['prob_ruin']:>19.1%} {results_b['prob_ruin']:>17.1%}")
print(f"{'Median Sharpe':.<30} {results_a['median_sharpe']:>20.3f} {results_b['median_sharpe']:>18.3f}")
print()

def recommend(r: dict) -> str:
    if r["prob_ruin"] > 0.05:
        return "SKIP — ruin prob too high"
    if r["median_sharpe"] < 0.5:
        return "SKIP — Sharpe too low"
    if r["win_rate"] < 0.60:
        return "CAUTION — borderline win rate"
    return "TRADE — validated edge"

print(f"  Cat A recommendation: {recommend(results_a)}")
print(f"  Cat B recommendation: {recommend(results_b)}")

# ── Combined strategy — optimized ─────────────────────────────────────────────
COMBINED_POSITION_SIZE  = 10.0
COMBINED_TRADES_PER_SIM = 200
COMBINED_DOUBLE         = STARTING_BANKROLL * 2  # $160

print("\n" + "=" * 60)
print("COMBINED STRATEGY — OPTIMIZED")
print("=" * 60)

# Combine both categories, filter YES price to 5–15c only
combined = pl.concat([cat_a_priced, cat_b_priced]).filter(
    pl.col("yes_price_cents") <= 15
)

c_n        = len(combined)
c_win_rate = combined["signal_correct"].mean()
c_avg_yes  = combined["yes_price_cents"].mean()
# no_cost = 1 - yes_price (as fraction); payout_ratio = yes_price / no_cost
c_payouts  = combined["payout_ratio"].to_numpy()

print(f"  Total qualifying signals : {c_n:,}")
print(f"  Win rate                 : {c_win_rate:.1%}")
print(f"  Avg YES price            : {c_avg_yes:.1f}c")

if c_n < 10:
    print("\nNot enough signals. Exiting combined section.")
    raise SystemExit(0)

rng = np.random.default_rng(seed=42)

c_paths    = np.zeros((N_SIMULATIONS, COMBINED_TRADES_PER_SIM + 1))
c_paths[:, 0] = STARTING_BANKROLL

for sim in range(N_SIMULATIONS):
    idx      = rng.integers(0, c_n, size=COMBINED_TRADES_PER_SIM)
    payouts  = c_payouts[idx]
    outcomes = rng.random(COMBINED_TRADES_PER_SIM) < c_win_rate
    bankroll = STARTING_BANKROLL
    for t in range(COMBINED_TRADES_PER_SIM):
        bankroll += COMBINED_POSITION_SIZE * payouts[t] if outcomes[t] else -COMBINED_POSITION_SIZE
        c_paths[sim, t + 1] = bankroll

c_final      = c_paths[:, -1]
c_prob_ruin  = float(np.mean(np.any(c_paths < RUIN_THRESHOLD, axis=1)))
c_prob_dbl   = float(np.mean(c_final >= COMBINED_DOUBLE))

def _max_dd(path):
    peak = np.maximum.accumulate(path)
    return float((peak - path).max())

c_drawdowns   = np.array([_max_dd(c_paths[i]) for i in range(N_SIMULATIONS)])
c_median_dd   = float(np.median(c_drawdowns))

# Sharpe
c_sharpes = np.zeros(N_SIMULATIONS)
for sim in range(N_SIMULATIONS):
    idx      = rng.integers(0, c_n, size=COMBINED_TRADES_PER_SIM)
    payouts  = c_payouts[idx]
    outcomes = rng.random(COMBINED_TRADES_PER_SIM) < c_win_rate
    returns  = np.where(outcomes, payouts, -1.0)
    std_r    = returns.std()
    c_sharpes[sim] = returns.mean() / std_r if std_r > 0 else 0.0

c_median_sharpe = float(np.median(c_sharpes))

print(f"\n  Monte Carlo ({N_SIMULATIONS:,} sims × {COMBINED_TRADES_PER_SIM} trades, ${COMBINED_POSITION_SIZE:.0f}/trade):")
print(f"    Median final bankroll  : ${np.median(c_final):.2f}")
print(f"    5th pct final bankroll : ${np.percentile(c_final, 5):.2f}  (worst case)")
print(f"    95th pct final bankroll: ${np.percentile(c_final, 95):.2f}  (best case)")
print(f"    Probability of ruin    : {c_prob_ruin:.1%}  (drops below ${RUIN_THRESHOLD:.0f})")
print(f"    Probability of doubling: {c_prob_dbl:.1%}  (reaches ${COMBINED_DOUBLE:.0f}+)")
print(f"    Median max drawdown    : ${c_median_dd:.2f}")
print(f"    Median Sharpe ratio    : {c_median_sharpe:.3f}")

print(f"\n  Position size sensitivity ($5–$20/trade, ruin threshold ${RUIN_THRESHOLD:.0f}):")
ref_size = COMBINED_POSITION_SIZE
for test_size in range(5, 21):
    scale        = test_size / ref_size
    scaled_paths = (c_paths - STARTING_BANKROLL) * scale + STARTING_BANKROLL
    ruin_p       = float(np.mean(np.any(scaled_paths < RUIN_THRESHOLD, axis=1)))
    median_br    = float(np.median(scaled_paths[:, -1]))
    marker       = " <-- current" if test_size == int(ref_size) else ""
    print(f"    ${test_size:>2}/trade  →  ruin {ruin_p:.1%}  |  median ${median_br:.0f}{marker}")
