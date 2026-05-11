"""Compare ensemble weather model accuracy against ERA5 observed high temperatures.

Tests 7 models across 6 cities and ~93 days (2026-02-04 to 2026-05-07).
Results cached to research/model_accuracy_comparison.parquet.
"""
from __future__ import annotations

import asyncio
import logging
import math
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = [
    {"name": "icon_seamless",            "label": "DWD ICON EPS",       "members": 39},
    {"name": "gfs_seamless",             "label": "NOAA GFS",            "members": 30},
    {"name": "ecmwf_aifs025",            "label": "ECMWF AIFS 0.25",    "members": 50},
    {"name": "ecmwf_ifs025",             "label": "ECMWF IFS 0.25",     "members": 50},
    {"name": "bom_access_global_ensemble","label": "BOM ACCESS",         "members": 17},
    {"name": "ncep_gefs025",             "label": "NOAA GEFS 0.25",     "members": 30},
    {"name": "ncep_gefs05",              "label": "NOAA GEFS 0.5",      "members": 30},
]

CITIES: Dict[str, Dict] = {
    "nyc":          {"lat": 40.7833, "lon": -73.9667, "name": "New York"},
    "chicago":      {"lat": 41.7842, "lon": -87.7553, "name": "Chicago"},
    "miami":        {"lat": 25.7906, "lon": -80.3164, "name": "Miami"},
    "los_angeles":  {"lat": 33.9381, "lon": -118.3889, "name": "Los Angeles"},
    "denver":       {"lat": 39.8466, "lon": -104.6562, "name": "Denver"},
    "philadelphia": {"lat": 39.8733, "lon": -75.2268, "name": "Philadelphia"},
}

START_DATE = date(2026, 2, 4)
END_DATE   = date(2026, 5, 7)
CACHE_PATH = Path(__file__).parent / "model_accuracy_comparison.parquet"

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"


def _date_range(start: date, end: date) -> List[date]:
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


async def _get_json(client: httpx.AsyncClient, url: str, params: dict, sem: asyncio.Semaphore) -> Optional[dict]:
    async with sem:
        for attempt in range(4):  # initial try + 3 retries
            try:
                r = await client.get(url, params=params, timeout=30.0)
                if r.status_code == 429:
                    retry_after = float(r.headers.get("Retry-After", 5))
                    logger.warning("429 rate limited — waiting %.0fs (attempt %d)", retry_after, attempt + 1)
                    await asyncio.sleep(retry_after)
                    continue
                r.raise_for_status()
                await asyncio.sleep(0.5)
                return r.json()
            except httpx.HTTPStatusError:
                raise
            except Exception as exc:
                if attempt == 0:
                    logger.warning("Request failed %s %s: %s — retrying in 2s", url, params.get("models", ""), exc)
                    await asyncio.sleep(2.0)
                else:
                    logger.warning("Request failed after retry %s %s: %s", url, params.get("models", ""), exc)
                    return None
        logger.warning("Exhausted retries for %s %s", url, params.get("models", ""))
        return None


async def fetch_actual_highs(
    client: httpx.AsyncClient,
    city_key: str,
    city: Dict,
    sem: asyncio.Semaphore,
) -> Dict[str, float]:
    """Fetch ERA5 actual daily high temps for all dates in range. Returns {date_str: high_f}."""
    params = {
        "latitude":          city["lat"],
        "longitude":         city["lon"],
        "daily":             "temperature_2m_max",
        "temperature_unit":  "fahrenheit",
        "timezone":          "America/New_York",
        "start_date":        START_DATE.isoformat(),
        "end_date":          END_DATE.isoformat(),
    }
    data = await _get_json(client, ARCHIVE_URL, params, sem)
    if not data or "daily" not in data:
        logger.warning("No archive data for %s", city_key)
        return {}
    dates  = data["daily"].get("time", [])
    temps  = data["daily"].get("temperature_2m_max", [])
    result: Dict[str, float] = {}
    for d, t in zip(dates, temps):
        if t is not None:
            result[d] = float(t)
    logger.info("Fetched %d actual highs for %s", len(result), city_key)
    return result


async def fetch_ensemble_mean_high(
    client: httpx.AsyncClient,
    city: Dict,
    model_name: str,
    target_date: str,
    sem: asyncio.Semaphore,
) -> Optional[float]:
    """Fetch ensemble forecast for one city/model/date and return mean member high."""
    params = {
        "latitude":         city["lat"],
        "longitude":        city["lon"],
        "hourly":           "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone":         "America/New_York",
        "start_date":       target_date,
        "end_date":         target_date,
        "models":           model_name,
    }
    data = await _get_json(client, ENSEMBLE_URL, params, sem)
    if not data or "hourly" not in data:
        return None

    hourly = data["hourly"]
    member_highs: List[float] = []
    for key, values in hourly.items():
        if "member" not in key:
            continue
        if not isinstance(values, list) or not values:
            continue
        valids = [v for v in values if v is not None]
        if valids:
            member_highs.append(max(valids))

    if not member_highs:
        return None
    return sum(member_highs) / len(member_highs)


def _load_cache() -> pl.DataFrame:
    if CACHE_PATH.exists():
        try:
            return pl.read_parquet(CACHE_PATH)
        except Exception as exc:
            logger.warning("Cache read failed: %s — starting fresh", exc)
    schema = {
        "model":      pl.Utf8,
        "city":       pl.Utf8,
        "date":       pl.Utf8,
        "model_mean": pl.Float64,
        "actual_high":pl.Float64,
        "error":      pl.Float64,
        "abs_error":  pl.Float64,
    }
    return pl.DataFrame(schema=schema)


def _save_cache(df: pl.DataFrame) -> None:
    df.write_parquet(CACHE_PATH)


async def run_fetching() -> pl.DataFrame:
    sem = asyncio.Semaphore(1)
    df = _load_cache()
    existing: set = set()
    if len(df) > 0:
        for row in df.iter_rows(named=True):
            existing.add((row["model"], row["city"], row["date"]))

    all_dates = _date_range(START_DATE, END_DATE)

    async with httpx.AsyncClient() as client:
        # Fetch all actual highs upfront — one sequential call per city.
        actuals: Dict[str, Dict[str, float]] = {}
        for city_key, city in CITIES.items():
            actuals[city_key] = await fetch_actual_highs(client, city_key, city, sem)

        # Process one model at a time, one city at a time, dates sequentially.
        for model_info in MODELS:
            model_name  = model_info["name"]
            model_label = model_info["label"]
            logger.info("=== Model: %s ===", model_label)

            model_rows: List[dict] = []
            completed = 0

            for city_key, city in CITIES.items():
                pending_dates = [
                    d.isoformat() for d in all_dates
                    if (model_name, city_key, d.isoformat()) not in existing
                    and actuals.get(city_key, {}).get(d.isoformat()) is not None
                ]
                if not pending_dates:
                    logger.info("  %s / %s: all cached", model_label, city_key)
                    continue

                logger.info("  %s / %s: %d dates to fetch", model_label, city_key, len(pending_dates))

                for date_str in pending_dates:
                    mean_high = await fetch_ensemble_mean_high(client, city, model_name, date_str, sem)
                    completed += 1
                    if mean_high is None:
                        continue
                    actual_high = actuals[city_key][date_str]
                    error = actual_high - mean_high
                    model_rows.append({
                        "model":       model_name,
                        "city":        city_key,
                        "date":        date_str,
                        "model_mean":  mean_high,
                        "actual_high": actual_high,
                        "error":       error,
                        "abs_error":   abs(error),
                    })
                    if completed % 20 == 0:
                        logger.info("  %s: %d done so far", model_label, completed)

            # Save after each model so partial progress is never lost.
            if model_rows:
                new_df = pl.DataFrame(model_rows)
                if len(df) > 0:
                    df = pl.concat([df, new_df])
                else:
                    df = new_df
                for row in model_rows:
                    existing.add((row["model"], row["city"], row["date"]))
                _save_cache(df)
                logger.info("  Saved %d new rows for %s (%d total)", len(model_rows), model_label, len(df))
            else:
                logger.info("  %s: nothing new to save", model_label)

    return df


def _rmse(errors: List[float]) -> float:
    if not errors:
        return float("nan")
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def _pct_within(abs_errors: List[float], threshold: float) -> float:
    if not abs_errors:
        return float("nan")
    return 100.0 * sum(1 for e in abs_errors if e <= threshold) / len(abs_errors)


def _direction_accuracy(rows: List[dict], distance: float) -> Optional[float]:
    """When |model_mean - threshold| >= distance, how often is the model correct?

    We simulate thresholds by treating the model_mean itself as the forecast and
    checking whether the direction call (above/below the midpoint between model
    and actual) would have been correct.  Practically: for each row where
    abs(model_mean - actual_high) >= distance, measure % where sign(error) == 0
    (i.e., actual matched the directional call).

    Concretely: the model says temp = M.  A hypothetical Kalshi threshold T sits
    at M ± distance.  The call is model_mean > T → YES.
      - For T = M - distance (model says YES, i.e. above T):
            correct if actual_high > T  = actual_high > M - distance
      - For T = M + distance (model says NO, i.e. below T):
            correct if actual_high < T  = actual_high < M + distance

    We average both cases to get symmetric direction accuracy.
    """
    correct = 0
    total   = 0
    for row in rows:
        m = row["model_mean"]
        a = row["actual_high"]
        # Case 1: threshold below model (model predicts above)
        t_below = m - distance
        total  += 1
        correct += 1 if a > t_below else 0
        # Case 2: threshold above model (model predicts below)
        t_above = m + distance
        total  += 1
        correct += 1 if a < t_above else 0
    if total == 0:
        return None
    return 100.0 * correct / total


def print_results(df: pl.DataFrame) -> None:
    if len(df) == 0:
        print("No data to analyze.")
        return

    model_stats: List[dict] = []

    print("\n" + "=" * 90)
    print("MODEL ACCURACY COMPARISON — 2026-02-04 to 2026-05-07")
    print("=" * 90)

    for model_info in MODELS:
        model_name  = model_info["name"]
        model_label = model_info["label"]

        sub = df.filter(pl.col("model") == model_name)
        if len(sub) == 0:
            print(f"\n{model_label}: no data")
            continue

        rows      = sub.to_dicts()
        abs_errs  = [r["abs_error"] for r in rows]
        errs      = [r["error"]     for r in rows]
        mae       = sum(abs_errs) / len(abs_errs)
        rmse_val  = _rmse(errs)
        bias      = sum(errs) / len(errs)
        w1f       = _pct_within(abs_errs, 1.0)
        w2f       = _pct_within(abs_errs, 2.0)
        w3f       = _pct_within(abs_errs, 3.0)
        n         = len(rows)

        print(f"\n{model_label} ({model_name})  n={n}")
        print(f"  MAE={mae:.2f}F  RMSE={rmse_val:.2f}F  Bias={bias:+.2f}F")
        print(f"  Within 1F: {w1f:.1f}%  2F: {w2f:.1f}%  3F: {w3f:.1f}%")

        dir_line = "  Direction accuracy:"
        for dist in [2, 3, 4, 5]:
            da = _direction_accuracy(rows, float(dist))
            if da is not None:
                dir_line += f"  @{dist}F={da:.1f}%"
        print(dir_line)

        # Per-city MAE
        city_maes = []
        for city_key, city_info in CITIES.items():
            csub = sub.filter(pl.col("city") == city_key)
            if len(csub) == 0:
                continue
            c_abs = csub["abs_error"].to_list()
            c_mae = sum(c_abs) / len(c_abs)
            city_maes.append((city_info["name"], c_mae, len(csub)))
        if city_maes:
            city_line = "  MAE by city:"
            for cname, cmae, cn in city_maes:
                city_line += f"  {cname}={cmae:.2f}F(n={cn})"
            print(city_line)

        model_stats.append({
            "label": model_label,
            "name":  model_name,
            "n":     n,
            "mae":   mae,
            "rmse":  rmse_val,
            "bias":  bias,
            "w1f":   w1f,
            "w2f":   w2f,
            "w3f":   w3f,
        })

    # Final ranking table
    model_stats.sort(key=lambda x: x["mae"])
    print("\n" + "=" * 90)
    print(f"{'RANK':<5} {'MODEL':<26} {'N':>6} {'MAE':>7} {'RMSE':>7} {'BIAS':>7} {'W1F':>7} {'W2F':>7} {'W3F':>7}")
    print("-" * 90)
    for rank, s in enumerate(model_stats, 1):
        print(
            f"{rank:<5} {s['label']:<26} {s['n']:>6} "
            f"{s['mae']:>7.2f} {s['rmse']:>7.2f} {s['bias']:>+7.2f} "
            f"{s['w1f']:>6.1f}% {s['w2f']:>6.1f}% {s['w3f']:>6.1f}%"
        )
    print("=" * 90)


async def main() -> None:
    df = await run_fetching()
    print_results(df)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    asyncio.run(main())
