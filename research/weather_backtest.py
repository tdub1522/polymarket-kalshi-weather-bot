"""
Weather contract backtest: Becker historical outcomes + Open-Meteo historical GFS ensemble.
For each resolved KXHIGH contract, fetch what the GFS ensemble was forecasting
and compare model probability vs actual outcome.
"""
import asyncio
import glob
import re
from datetime import date, datetime, timezone, timedelta
from typing import Optional
import httpx
import polars as pl

# ── Config ────────────────────────────────────────────────────────────────────
BECKER_PATH = "/Users/treywoolley/Desktop/prediction-market-analysis/data/kalshi/markets"
MIN_CONTRACTS = 20  # minimum contracts per signal bucket to include in results

CITY_COORDS = {
    "NY":   {"lat": 40.7128, "lon": -74.0060, "name": "New York"},
    "CHI":  {"lat": 41.8781, "lon": -87.6298, "name": "Chicago"},
    "MIA":  {"lat": 25.7617, "lon": -80.1918, "name": "Miami"},
    "LAX":  {"lat": 34.0522, "lon": -118.2437, "name": "Los Angeles"},
    "DEN":  {"lat": 39.7392, "lon": -104.9903, "name": "Denver"},
    "AUS":  {"lat": 30.2672, "lon": -97.7431, "name": "Austin"},
    "PHIL": {"lat": 39.9526, "lon": -75.1652, "name": "Philadelphia"},
    "HOU":  {"lat": 29.7604, "lon": -95.3698, "name": "Houston"},
}

MONTH_ABBR = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# ── Ticker Parser ─────────────────────────────────────────────────────────────
def parse_ticker(ticker: str) -> Optional[dict]:
    match = re.match(r'^KXHIGH([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})-([BT])([\d.]+)$', ticker)
    if not match:
        return None
    city = match.group(1)
    yy, mon_str, dd = int(match.group(2)), match.group(3), int(match.group(4))
    boundary = match.group(5)
    threshold = float(match.group(6))
    month = MONTH_ABBR.get(mon_str)
    if not month or city not in CITY_COORDS:
        return None
    try:
        target_date = date(2000 + yy, month, dd)
    except ValueError:
        return None
    direction = "above" if boundary == "B" else "below"
    return {
        "city": city,
        "target_date": target_date,
        "threshold_f": threshold,
        "direction": direction,
        "boundary": boundary,
    }

# ── Open-Meteo Historical Actual Fetch ────────────────────────────────────────
async def fetch_historical_actual(city: str, target_date: date) -> Optional[dict]:
    coords = CITY_COORDS[city]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        actual_high_f = float(data["daily"]["temperature_2m_max"][0])
        return {
            "actual_high_f": actual_high_f,
            "mean_high": actual_high_f,
            "std_high": 0.0,
            "num_members": 1,
            "member_highs": [actual_high_f],
        }
    except Exception as e:
        print(f"  [WARN] Historical fetch failed for {city} {target_date}: {e}")
        return None

# ── Model Probability Calculator ──────────────────────────────────────────────
def calc_model_prob(forecast: dict, threshold_f: float, direction: str) -> float:
    members = forecast["member_highs"]
    if direction == "above":
        low = threshold_f - 0.5
        high = threshold_f + 0.5
        prob = len([m for m in members if low <= m <= high]) / len(members)
    else:  # below
        prob = len([m for m in members if m < threshold_f]) / len(members)
    return max(0.02, min(0.98, prob))

# ── Main Backtest ─────────────────────────────────────────────────────────────
async def main():
    print("Loading Becker weather contracts...")
    files = glob.glob(f"{BECKER_PATH}/*.parquet")
    markets = pl.concat([pl.read_parquet(f) for f in files])

    weather = markets.filter(
        pl.col("ticker").str.contains("KXHIGH") &
        (pl.col("result") != "")
    ).select("ticker", "result", "close_time", "yes_ask", "no_ask")

    print(f"Found {len(weather)} resolved weather contracts")
    print("Fetching historical GFS ensemble data...\n")

    results = []
    seen_dates = {}  # cache: (city, date) -> forecast

    rows = weather.to_dicts()
    total = len(rows)

    for i, row in enumerate(rows):
        ticker = row["ticker"]
        parsed = parse_ticker(ticker)
        if not parsed:
            continue

        city = parsed["city"]
        target_date = parsed["target_date"]
        threshold_f = parsed["threshold_f"]
        direction = parsed["direction"]
        actual_result = row["result"]  # "yes" or "no"
        yes_ask = (row["yes_ask"] or 50) / 100.0
        no_ask = (row["no_ask"] or 50) / 100.0

        # Fetch ensemble (cached per city+date)
        cache_key = (city, target_date)
        if cache_key not in seen_dates:
            if i % 50 == 0:
                print(f"Progress: {i}/{total} contracts processed...")
            forecast = await fetch_historical_actual(city, target_date)
            seen_dates[cache_key] = forecast
        else:
            forecast = seen_dates[cache_key]

        if not forecast:
            continue

        model_prob = calc_model_prob(forecast, threshold_f, direction)
        market_prob = yes_ask
        edge = model_prob - market_prob
        mean_high = forecast["mean_high"]
        std_high = forecast["std_high"]

        # Did our signal win?
        if edge > 0:
            signal_side = "yes"
            signal_correct = actual_result == "yes"
        else:
            signal_side = "no"
            signal_correct = actual_result == "no"

        results.append({
            "ticker": ticker,
            "city": city,
            "target_date": target_date.isoformat(),
            "threshold_f": threshold_f,
            "direction": direction,
            "model_prob": round(model_prob, 4),
            "market_prob": round(market_prob, 4),
            "edge": round(edge, 4),
            "mean_high": round(mean_high, 2),
            "std_high": round(std_high, 2),
            "num_members": forecast["num_members"],
            "actual_result": actual_result,
            "signal_side": signal_side,
            "signal_correct": signal_correct,
        })

    df = pl.DataFrame(results)
    output_path = "/Users/treywoolley/Desktop/kalshi-bot/polymarket-kalshi-weather-bot/research/backtest_results.parquet"
    df.write_parquet(output_path)
    print(f"\nSaved {len(df)} results to {output_path}")

    # ── Summary Stats ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)

    # Win rate by edge bucket
    df = df.with_columns([
        (pl.col("edge").abs() * 10).cast(pl.Int32).alias("edge_bucket")
    ])

    actionable = df.filter(pl.col("edge").abs() >= 0.08)
    print(f"\nTotal signals with edge >= 8%: {len(actionable)}")
    print(f"Win rate: {actionable['signal_correct'].mean():.1%}")

    print("\nWin rate by edge bucket (>= 8% edge):")
    summary = (
        actionable
        .group_by("edge_bucket")
        .agg([
            pl.len().alias("count"),
            pl.col("signal_correct").mean().alias("win_rate"),
            pl.col("edge").mean().alias("avg_edge"),
        ])
        .sort("edge_bucket")
    )
    print(summary)

    print("\nWin rate by city:")
    by_city = (
        actionable
        .group_by("city")
        .agg([
            pl.len().alias("count"),
            pl.col("signal_correct").mean().alias("win_rate"),
        ])
        .sort("win_rate", descending=True)
    )
    print(by_city)

if __name__ == "__main__":
    asyncio.run(main())
