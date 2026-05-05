"""
GFS directional accuracy analysis for T-contracts (direction == "below").
Measures how well the GFS mean_high predicts the actual YES/NO outcome.
"""
import polars as pl
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "backtest_results.parquet"
MIN_SWEET_SPOT_CONTRACTS = 500

# ── Load ──────────────────────────────────────────────────────────────────────
df = pl.read_parquet(RESULTS_PATH)

# Keep only T-contracts
t_contracts = df.filter(pl.col("direction") == "below")
print(f"Total T-contracts (direction=below): {len(t_contracts)}")

# ── Feature columns ───────────────────────────────────────────────────────────
t_contracts = t_contracts.with_columns([
    pl.when(pl.col("mean_high") > pl.col("threshold_f"))
      .then(pl.lit("no"))
      .otherwise(pl.lit("yes"))
      .alias("gfs_prediction"),

    (pl.col("mean_high") - pl.col("threshold_f")).abs().alias("distance"),
])

t_contracts = t_contracts.with_columns(
    (pl.col("gfs_prediction") == pl.col("actual_result")).alias("gfs_correct")
)

# ── 1. Overall accuracy ───────────────────────────────────────────────────────
total      = len(t_contracts)
accuracy   = t_contracts["gfs_correct"].mean()

pred_no    = t_contracts.filter(pl.col("gfs_prediction") == "no")
pred_yes   = t_contracts.filter(pl.col("gfs_prediction") == "yes")
acc_no     = pred_no["gfs_correct"].mean() if len(pred_no) else float("nan")
acc_yes    = pred_yes["gfs_correct"].mean() if len(pred_yes) else float("nan")

print("\n" + "=" * 60)
print("OVERALL GFS DIRECTIONAL ACCURACY")
print("=" * 60)
print(f"  Total contracts      : {total:,}")
print(f"  Overall accuracy     : {accuracy:.1%}")
print(f"  Accuracy (pred NO)   : {acc_no:.1%}  ({len(pred_no):,} contracts)")
print(f"  Accuracy (pred YES)  : {acc_yes:.1%}  ({len(pred_yes):,} contracts)")

# ── 2. Accuracy by distance bucket ────────────────────────────────────────────
print("\n" + "=" * 60)
print("GFS ACCURACY BY DISTANCE (mean_high vs threshold_f)")
print("=" * 60)

bucketed = t_contracts.with_columns(
    pl.when(pl.col("distance") < 1).then(pl.lit("0-1F"))
    .when(pl.col("distance") < 2).then(pl.lit("1-2F"))
    .when(pl.col("distance") < 3).then(pl.lit("2-3F"))
    .when(pl.col("distance") < 5).then(pl.lit("3-5F"))
    .when(pl.col("distance") < 8).then(pl.lit("5-8F"))
    .otherwise(pl.lit("8F+"))
    .alias("dist_bucket")
)

by_distance = (
    bucketed
    .group_by("dist_bucket")
    .agg([
        pl.len().alias("count"),
        pl.col("gfs_correct").mean().alias("accuracy"),
    ])
    .with_columns(
        (pl.col("count") / total * 100).alias("pct_of_total")
    )
    .sort("dist_bucket")
)
print(by_distance)

# ── 3. Accuracy by city ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GFS ACCURACY BY CITY")
print("=" * 60)

by_city = (
    t_contracts
    .group_by("city")
    .agg([
        pl.len().alias("count"),
        pl.col("gfs_correct").mean().alias("accuracy"),
    ])
    .sort("accuracy", descending=True)
)
print(by_city)

# ── 4. Confusion matrix ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

tn = len(t_contracts.filter((pl.col("gfs_prediction") == "no")  & (pl.col("actual_result") == "no")))
fn = len(t_contracts.filter((pl.col("gfs_prediction") == "no")  & (pl.col("actual_result") == "yes")))
tp = len(t_contracts.filter((pl.col("gfs_prediction") == "yes") & (pl.col("actual_result") == "yes")))
fp = len(t_contracts.filter((pl.col("gfs_prediction") == "yes") & (pl.col("actual_result") == "no")))

print(f"  GFS pred NO  + actual NO  (true negative)  : {tn:>5,}  ({tn/total:.1%})")
print(f"  GFS pred NO  + actual YES (false negative)  : {fn:>5,}  ({fn/total:.1%})")
print(f"  GFS pred YES + actual YES (true positive)   : {tp:>5,}  ({tp/total:.1%})")
print(f"  GFS pred YES + actual NO  (false positive)  : {fp:>5,}  ({fp/total:.1%})")
print(f"  Total: {tn + fn + tp + fp:,}")

# ── 5. GFS NO predictions only ───────────────────────────────────────────────
MIN_NO_CONTRACTS = 200

gfs_no = t_contracts.filter(pl.col("gfs_prediction") == "no")
no_total    = len(gfs_no)
no_accuracy = gfs_no["gfs_correct"].mean()

print("\n" + "=" * 60)
print("GFS NO PREDICTIONS ONLY  (mean_high > threshold_f)")
print("=" * 60)
print(f"  Total NO predictions : {no_total:,}")
print(f"  Overall accuracy     : {no_accuracy:.1%}")

print("\n  Accuracy by distance bucket:")
no_bucketed = gfs_no.with_columns(
    pl.when(pl.col("distance") < 1).then(pl.lit("0-1F"))
    .when(pl.col("distance") < 2).then(pl.lit("1-2F"))
    .when(pl.col("distance") < 3).then(pl.lit("2-3F"))
    .when(pl.col("distance") < 5).then(pl.lit("3-5F"))
    .when(pl.col("distance") < 8).then(pl.lit("5-8F"))
    .otherwise(pl.lit("8F+"))
    .alias("dist_bucket")
)
no_by_distance = (
    no_bucketed
    .group_by("dist_bucket")
    .agg([
        pl.len().alias("count"),
        pl.col("gfs_correct").mean().alias("accuracy"),
    ])
    .sort("dist_bucket")
)
print(no_by_distance)

print("\n  Accuracy by city:")
no_by_city = (
    gfs_no
    .group_by("city")
    .agg([
        pl.len().alias("count"),
        pl.col("gfs_correct").mean().alias("accuracy"),
    ])
    .sort("accuracy", descending=True)
)
print(no_by_city)

print(f"\n  Sweet spot (0.0–5.0F in 0.5F steps, >= {MIN_NO_CONTRACTS} contracts):")
no_best_threshold = None
no_best_accuracy  = 0.0
for threshold in [d / 2 for d in range(0, 11)]:  # 0.0 to 5.0 in 0.5F steps
    subset = gfs_no.filter(pl.col("distance") >= threshold)
    n = len(subset)
    if n < MIN_NO_CONTRACTS:
        break
    acc = subset["gfs_correct"].mean()
    if acc > no_best_accuracy:
        no_best_accuracy  = acc
        no_best_threshold = threshold
    marker = " <-- best" if threshold == no_best_threshold else ""
    print(f"    >= {threshold:.1f}F  →  {acc:.1%}  ({n:,} contracts){marker}")

# ── 6. Sweet spot: best distance threshold ────────────────────────────────────
print("\n" + "=" * 60)
print(f"OVERALL SWEET SPOT  (max accuracy with >= {MIN_SWEET_SPOT_CONTRACTS} contracts)")
print("=" * 60)

best_threshold = None
best_accuracy  = 0.0
best_count     = 0

for threshold in [d / 10 for d in range(0, 101)]:  # 0.0 to 10.0 in 0.1F steps
    subset = t_contracts.filter(pl.col("distance") >= threshold)
    n = len(subset)
    if n < MIN_SWEET_SPOT_CONTRACTS:
        break
    acc = subset["gfs_correct"].mean()
    if acc > best_accuracy:
        best_accuracy  = acc
        best_threshold = threshold
        best_count     = n

if best_threshold is not None:
    print(f"  Best distance threshold : >= {best_threshold:.1f}F")
    print(f"  Accuracy at threshold   : {best_accuracy:.1%}")
    print(f"  Contracts remaining     : {best_count:,}")
    print()
    print("  Accuracy vs distance threshold (0.5F steps, >= 500 contracts):")
    for threshold in [d / 2 for d in range(0, 21)]:  # 0.0 to 10.0 in 0.5F steps
        subset = t_contracts.filter(pl.col("distance") >= threshold)
        n = len(subset)
        if n < MIN_SWEET_SPOT_CONTRACTS:
            break
        acc = subset["gfs_correct"].mean()
        marker = " <-- best" if threshold == best_threshold else ""
        print(f"    >= {threshold:.1f}F  →  {acc:.1%}  ({n:,} contracts){marker}")
else:
    print("  No threshold found with enough contracts.")
