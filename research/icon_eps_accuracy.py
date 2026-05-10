"""ICON deterministic forecast accuracy vs actual observed temperatures.

Fetches ICON seamless daily high forecasts and ERA5 observed highs from the
Open-Meteo archive API, then validates directional accuracy by distance bucket.
"""
import asyncio
import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import httpx
import polars as pl

CITIES = {
    "nyc":          {"lat": 40.7833,  "lon":  -73.9667, "name": "New York City"},
    "chicago":      {"lat": 41.7842,  "lon":  -87.7553, "name": "Chicago"},
    "miami":        {"lat": 25.7906,  "lon":  -80.3164, "name": "Miami"},
    "los_angeles":  {"lat": 33.9381,  "lon": -118.3889, "name": "Los Angeles"},
    "denver":       {"lat": 39.8466,  "lon": -104.6562, "name": "Denver"},
    "philadelphia": {"lat": 39.8733,  "lon":  -75.2268, "name": "Philadelphia"},
}

START_DATE = date(2026, 2, 4)
END_DATE   = date(2026, 5, 7)
CACHE_PATH = Path(__file__).parent / "icon_eps_validation.parquet"


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


async def fetch_archive_high(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    target_date: date,
    sem: asyncio.Semaphore,
    model: Optional[str] = None,
) -> Optional[float]:
    """Fetch daily max temperature (°F) from Open-Meteo archive.

    Pass model="icon_seamless" for ICON forecast; omit for ERA5 observed.
    """
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "daily":            "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone":         "America/New_York",
        "start_date":       target_date.isoformat(),
        "end_date":         target_date.isoformat(),
    }
    if model:
        params["models"] = model

    async with sem:
        resp = await client.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()

    highs = data.get("daily", {}).get("temperature_2m_max", [])
    if highs and highs[0] is not None:
        return float(highs[0])
    return None


async def fetch_row(
    client: httpx.AsyncClient,
    city_key: str,
    city: dict,
    target_date: date,
    sem: asyncio.Semaphore,
) -> Optional[dict]:
    try:
        icon_forecast, actual_high = await asyncio.gather(
            fetch_archive_high(client, city["lat"], city["lon"], target_date, sem, model="icon_seamless"),
            fetch_archive_high(client, city["lat"], city["lon"], target_date, sem, model=None),
        )
        if icon_forecast is None or actual_high is None:
            return None
        error = actual_high - icon_forecast
        return {
            "city":          city_key,
            "city_name":     city["name"],
            "date":          target_date.isoformat(),
            "icon_forecast": icon_forecast,
            "actual_high":   actual_high,
            "error":         error,
            "abs_error":     abs(error),
        }
    except Exception as e:
        print(f"  ERROR {city_key} {target_date}: {e}")
        return None


async def main():
    # ── Load cache ────────────────────────────────────────────────────────
    if CACHE_PATH.exists():
        df_cache = pl.read_parquet(CACHE_PATH)
        cached_keys = set(zip(df_cache["city"].to_list(), df_cache["date"].to_list()))
        print(f"Loaded {len(df_cache)} cached rows from {CACHE_PATH.name}")
    else:
        df_cache = None
        cached_keys = set()

    # ── Determine missing pairs ───────────────────────────────────────────
    all_dates = list(date_range(START_DATE, END_DATE))
    missing: list[Tuple[str, date]] = [
        (city_key, d)
        for d in all_dates
        for city_key in CITIES
        if (city_key, d.isoformat()) not in cached_keys
    ]
    print(f"Need to fetch: {len(missing)} rows ({len(all_dates)} dates × {len(CITIES)} cities)")

    # ── Fetch missing rows ────────────────────────────────────────────────
    new_rows: list[dict] = []
    if missing:
        sem = asyncio.Semaphore(5)
        async with httpx.AsyncClient() as client:
            dates_seen: set[str] = set()
            for city_key, target_date in missing:
                date_str = target_date.isoformat()
                if date_str not in dates_seen:
                    dates_seen.add(date_str)
                    if len(dates_seen) % 10 == 0:
                        print(f"  Progress: {len(dates_seen)} dates processed, {len(new_rows)} rows fetched")

                row = await fetch_row(client, city_key, CITIES[city_key], target_date, sem)
                if row:
                    new_rows.append(row)

        print(f"Fetched {len(new_rows)} new rows")

        df_new = pl.DataFrame(new_rows).with_columns([
            pl.col("icon_forecast").cast(pl.Float64),
            pl.col("actual_high").cast(pl.Float64),
            pl.col("error").cast(pl.Float64),
            pl.col("abs_error").cast(pl.Float64),
        ])
        df = pl.concat([df_cache, df_new]) if df_cache is not None else df_new
        df.write_parquet(CACHE_PATH)
        print(f"Saved {len(df)} total rows to {CACHE_PATH.name}")
    else:
        df = df_cache
        print("Cache is complete — no fetching needed")

    if df is None or len(df) == 0:
        print("No data to analyze.")
        return

    # ── Analysis ──────────────────────────────────────────────────────────
    total = len(df)
    mae   = df["abs_error"].mean()
    rmse  = (df.select((pl.col("error") ** 2).mean()).item()) ** 0.5
    bias  = df["error"].mean()

    print("\n" + "=" * 60)
    print("ICON DETERMINISTIC FORECAST ACCURACY")
    print(f"Date range : {START_DATE} → {END_DATE}  ({total} rows)")
    print("=" * 60)
    print(f"\nOverall MAE  : {mae:.2f}°F")
    print(f"Overall RMSE : {rmse:.2f}°F")
    print(f"Overall Bias : {bias:+.2f}°F  (+ = ICON runs cold, − = runs warm)")

    # ── Per-city MAE / RMSE / Bias ────────────────────────────────────────
    print("\n── MAE / RMSE / Bias by City ──────────────────────────────")
    city_stats = (
        df.group_by("city_name")
        .agg([
            pl.col("abs_error").mean().alias("mae"),
            ((pl.col("error") ** 2).mean() ** 0.5).alias("rmse"),
            pl.col("error").mean().alias("bias"),
            pl.col("error").count().alias("n"),
        ])
        .sort("mae")
    )
    print(f"{'City':<22} {'N':>5}  {'MAE':>6}  {'RMSE':>6}  {'Bias':>7}")
    print("-" * 54)
    for row in city_stats.iter_rows(named=True):
        print(
            f"{row['city_name']:<22} {row['n']:>5}  "
            f"{row['mae']:>5.2f}F  {row['rmse']:>5.2f}F  {row['bias']:>+6.2f}F"
        )

    # ── % forecasts within tolerance ─────────────────────────────────────
    print("\n── Forecast Accuracy Within Tolerance ─────────────────────")
    for tol in [1.0, 2.0, 3.0]:
        within = len(df.filter(pl.col("abs_error") <= tol))
        print(f"  Within {tol:.0f}F : {within:>4}/{total}  ({within/total:.1%})")

    # ── Direction accuracy by distance from threshold ─────────────────────
    #
    # Model: ICON forecast is d°F above a hypothetical threshold T.
    #   icon_forecast = T + d  →  T = icon_forecast − d
    #   NO bet wins when actual_high > T = actual_high > icon_forecast − d = error > −d
    #   P(win | distance = d) = P(error > −d)
    #
    print("\n── Direction Accuracy by Distance from Threshold ──────────")
    print("(ICON forecast d°F above threshold; correct = actual also above threshold)")
    print(f"{'Distance':>10}  {'Correct':>9}  {'Win Rate':>9}  {'Mean Error':>11}")
    print("-" * 48)
    for d in [1.0, 2.0, 3.0, 4.0, 5.0]:
        correct = len(df.filter(pl.col("error") > -d))
        win_rate = correct / total
        mean_err = df.filter(pl.col("error") > -d)["error"].mean()
        print(
            f"At {d:.0f}F ICON distance, direction correct {win_rate:.1%} of time  "
            f"({correct}/{total}, mean error {mean_err:+.2f}F)"
        )

    # ── Absolute error distribution ───────────────────────────────────────
    print("\n── Absolute Error Distribution ─────────────────────────────")
    error_bands = [
        ("< 1F",  0.0, 1.0),
        ("1–2F",  1.0, 2.0),
        ("2–3F",  2.0, 3.0),
        ("3–5F",  3.0, 5.0),
        ("≥ 5F",  5.0, float("inf")),
    ]
    for label, lo, hi in error_bands:
        if hi == float("inf"):
            count = len(df.filter(pl.col("abs_error") >= lo))
        else:
            count = len(df.filter((pl.col("abs_error") >= lo) & (pl.col("abs_error") < hi)))
        bar = "█" * int(count / total * 40)
        print(f"  |error| {label:<6}: {count:>4}/{total}  {count/total:>5.1%}  {bar}")


if __name__ == "__main__":
    asyncio.run(main())
