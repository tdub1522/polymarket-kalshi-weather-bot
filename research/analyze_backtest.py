"""
Analyze backtest_results.parquet — win rate breakdowns for weather signal calibration.
All analyses filtered to actionable signals where abs(edge) >= 0.08.
"""
import polars as pl
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "backtest_results.parquet"
MIN_EDGE = 0.08
MIN_COMBO_CONTRACTS = 50

df = pl.read_parquet(RESULTS_PATH)
actionable = df.filter(pl.col("edge").abs() >= MIN_EDGE)
print(f"Loaded {len(df)} total rows, {len(actionable)} actionable (|edge| >= {MIN_EDGE:.0%})\n")


# ── 1. Win rate by GFS mean_high distance from threshold ──────────────────────
print("=" * 60)
print("1. WIN RATE BY |mean_high - threshold_f| DISTANCE")
print("=" * 60)

distance_buckets = (
    actionable
    .with_columns(
        (pl.col("mean_high") - pl.col("threshold_f")).abs().alias("dist")
    )
    .with_columns(
        pl.when(pl.col("dist") < 1).then(pl.lit("0-1F"))
        .when(pl.col("dist") < 2).then(pl.lit("1-2F"))
        .when(pl.col("dist") < 3).then(pl.lit("2-3F"))
        .when(pl.col("dist") < 5).then(pl.lit("3-5F"))
        .when(pl.col("dist") < 8).then(pl.lit("5-8F"))
        .otherwise(pl.lit("8F+"))
        .alias("dist_bucket")
    )
    .group_by("dist_bucket")
    .agg([
        pl.len().alias("count"),
        pl.col("signal_correct").mean().alias("win_rate"),
    ])
    .sort("dist_bucket")
)
print(distance_buckets)


# ── 2. Win rate by model_prob ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. WIN RATE BY MODEL PROBABILITY")
print("=" * 60)

model_prob_buckets = (
    actionable
    .with_columns(
        pl.when(pl.col("model_prob") < 0.10).then(pl.lit("0-10%"))
        .when(pl.col("model_prob") < 0.20).then(pl.lit("10-20%"))
        .when(pl.col("model_prob") < 0.30).then(pl.lit("20-30%"))
        .when(pl.col("model_prob") < 0.70).then(pl.lit("30-70%"))
        .when(pl.col("model_prob") < 0.80).then(pl.lit("70-80%"))
        .when(pl.col("model_prob") < 0.90).then(pl.lit("80-90%"))
        .otherwise(pl.lit("90-100%"))
        .alias("prob_bucket")
    )
    .group_by("prob_bucket")
    .agg([
        pl.len().alias("count"),
        pl.col("signal_correct").mean().alias("win_rate"),
    ])
    .sort("prob_bucket")
)
print(model_prob_buckets)


# ── 3. Win rate by market_prob (cents) ────────────────────────────────────────
print("\n" + "=" * 60)
print("3. WIN RATE BY MARKET PROBABILITY (CENTS)")
print("=" * 60)

market_prob_buckets = (
    actionable
    .with_columns(
        (pl.col("market_prob") * 100).alias("cents")
    )
    .with_columns(
        pl.when(pl.col("cents") <= 5).then(pl.lit("1-5c"))
        .when(pl.col("cents") <= 10).then(pl.lit("6-10c"))
        .when(pl.col("cents") <= 20).then(pl.lit("11-20c"))
        .when(pl.col("cents") <= 30).then(pl.lit("21-30c"))
        .when(pl.col("cents") <= 40).then(pl.lit("31-40c"))
        .when(pl.col("cents") <= 80).then(pl.lit("40-80c"))
        .when(pl.col("cents") <= 90).then(pl.lit("80-90c"))
        .otherwise(pl.lit("90-99c"))
        .alias("cents_bucket")
    )
    .group_by("cents_bucket")
    .agg([
        pl.len().alias("count"),
        pl.col("signal_correct").mean().alias("win_rate"),
    ])
    .sort("cents_bucket")
)
print(market_prob_buckets)


# ── 4. Win rate by direction ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. WIN RATE BY DIRECTION (above vs below)")
print("=" * 60)

by_direction = (
    actionable
    .group_by("direction")
    .agg([
        pl.len().alias("count"),
        pl.col("signal_correct").mean().alias("win_rate"),
    ])
    .sort("direction")
)
print(by_direction)


# ── 5. Win rate by signal_side ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. WIN RATE BY SIGNAL SIDE (yes vs no)")
print("=" * 60)

by_side = (
    actionable
    .group_by("signal_side")
    .agg([
        pl.len().alias("count"),
        pl.col("signal_correct").mean().alias("win_rate"),
    ])
    .sort("signal_side")
)
print(by_side)


# ── 6. Best city + direction + signal_side combinations ───────────────────────
print("\n" + "=" * 60)
print(f"6. BEST CITY + DIRECTION + SIGNAL_SIDE COMBOS (min {MIN_COMBO_CONTRACTS} contracts)")
print("=" * 60)

combos = (
    actionable
    .group_by(["city", "direction", "signal_side"])
    .agg([
        pl.len().alias("count"),
        pl.col("signal_correct").mean().alias("win_rate"),
        pl.col("edge").abs().mean().alias("avg_edge"),
    ])
    .filter(pl.col("count") >= MIN_COMBO_CONTRACTS)
    .sort("win_rate", descending=True)
)
print(combos)
