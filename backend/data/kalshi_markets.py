"""Kalshi weather temperature market fetcher."""
import asyncio
import logging
import re
import time
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from backend.data.kalshi_client import KalshiClient, kalshi_credentials_present
from backend.data.weather_markets import WeatherMarket

logger = logging.getLogger("trading_bot")

# Kalshi series tickers for high-temperature markets by city
CITY_SERIES: Dict[str, str] = {
    "nyc":           "KXHIGHNY",
    "chicago":       "KXHIGHCHI",
    "miami":         "KXHIGHMIA",
    "los_angeles":   "KXHIGHLAX",
    "denver":        "KXHIGHDEN",
    "boston":        "KXHIGHBOS",
    "philadelphia":  "KXHIGHPHIL",
    "atlanta":       "KXHIGHATL",
    "san_francisco": "KXHIGHTSFO",
    "minneapolis":   "KXHIGHTMIN",
    "phoenix":       "KXHIGHTPHX",
    "houston":       "KXHIGHTHOU",
}

CITY_NAMES: Dict[str, str] = {
    "nyc":           "New York",
    "chicago":       "Chicago",
    "miami":         "Miami",
    "los_angeles":   "Los Angeles",
    "denver":        "Denver",
    "boston":        "Boston",
    "philadelphia":  "Philadelphia",
    "atlanta":       "Atlanta",
    "san_francisco": "San Francisco",
    "minneapolis":   "Minneapolis",
    "phoenix":       "Phoenix",
    "houston":       "Houston",
}

# Month abbreviation mapping for ticker parsing
MONTH_ABBR = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_kalshi_ticker(ticker: str) -> Optional[dict]:
    """
    Parse a Kalshi bracket ticker into market parameters.

    Format: KXHIGHNY-26MAR01-B45.5
      - 26MAR01 = 2026-03-01
      - B45.5 = bracket boundary at 45.5°F (above)
      - T45.5 would be "at or below" (top boundary)
    """
    # Match: SERIES-YYMONDD-B/Tnn.n
    match = re.match(
        r'^[A-Z]+-(\d{2})([A-Z]{3})(\d{2})-([BT])([\d.]+)$',
        ticker,
    )
    if not match:
        return None

    yy = int(match.group(1))
    mon_str = match.group(2)
    dd = int(match.group(3))
    boundary_type = match.group(4)
    threshold = float(match.group(5))

    month = MONTH_ABBR.get(mon_str)
    if not month:
        return None

    year = 2000 + yy
    try:
        target_date = date(year, month, dd)
    except ValueError:
        return None

    # B = bottom boundary → "above" threshold; T = top boundary → "below" threshold
    direction = "above" if boundary_type == "B" else "below"

    return {
        "target_date": target_date,
        "threshold_f": threshold,
        "metric": "high",
        "direction": direction,
    }


async def _fetch_market_detail(
    client: KalshiClient, ticker: str
) -> tuple[float, float, float]:
    """
    Fetch individual market detail for accurate bid/ask prices and volume.
    Returns (yes_price, no_price, volume). Falls back to (0.5, 0.5, 0.0) on error or timeout.
    """
    try:
        data = await asyncio.wait_for(client.get_market(ticker), timeout=5.0)
        m = data.get("market", {})

        yes_price = float(m.get("yes_ask_dollars") or 0)
        no_price = float(m.get("no_ask_dollars") or 0)

        # Fall back to bid if ask is missing
        if yes_price <= 0:
            yes_price = float(m.get("yes_bid_dollars") or 0)
        if no_price <= 0:
            no_price = float(m.get("no_bid_dollars") or 0)

        # Final fallback
        if yes_price <= 0:
            yes_price = 0.5
        if no_price <= 0:
            no_price = 0.5

        volume = float(m.get("volume_fp") or 0)
        return yes_price, no_price, volume

    except asyncio.TimeoutError:
        logger.info(f"PRICE FETCH TIMEOUT for {ticker} (>5s)")
        return 0.5, 0.5, 0.0
    except Exception as e:
        logger.info(f"PRICE FETCH ERROR for {ticker}: {type(e).__name__}: {e}")
        return 0.5, 0.5, 0.0


async def fetch_kalshi_weather_markets(
    city_keys: Optional[List[str]] = None,
) -> List[WeatherMarket]:
    """
    Fetch open weather temperature markets from Kalshi.

    Queries the KXHIGH{city} series for each configured city,
    handles cursor-based pagination, and returns WeatherMarket objects.
    """
    if not kalshi_credentials_present():
        return []

    client = KalshiClient()
    markets: List[WeatherMarket] = []
    today = date.today()

    cities = city_keys or list(CITY_SERIES.keys())

    # Phase 1: collect candidates across all cities (no price fetching yet)
    candidates = []  # list of dicts: ticker, city_key, city_name, parsed, raw title

    for city_key in cities:
        series = CITY_SERIES.get(city_key)
        if not series:
            continue

        city_name = CITY_NAMES.get(city_key, city_key)
        cursor = None

        try:
            while True:
                params = {
                    "series_ticker": series,
                    "status": "open",
                    "limit": 200,
                }
                if cursor:
                    params["cursor"] = cursor

                data = await client.get_markets(params)
                raw_markets = data.get("markets", [])

                for m in raw_markets:
                    ticker = m.get("ticker", "")
                    parsed = _parse_kalshi_ticker(ticker)
                    if not parsed:
                        continue

                    if parsed["target_date"] < today:
                        continue

                    close_time_str = m.get("close_time", "")
                    if close_time_str:
                        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                        now = datetime.now(timezone.utc)
                        hours_until_close = (close_time - now).total_seconds() / 3600
                        if hours_until_close > 48 or hours_until_close < 0:
                            continue
                        logger.info(f"{ticker} closes in {hours_until_close:.1f}h")

                    candidates.append({
                        "ticker": ticker,
                        "city_key": city_key,
                        "city_name": city_name,
                        "parsed": parsed,
                        "title": m.get("title", ticker),
                    })
                    logger.info(f"Candidate added: {ticker} closes in {hours_until_close:.1f}h")

                cursor = data.get("cursor")
                if not cursor or not raw_markets:
                    break

        except Exception as e:
            logger.warning(f"Failed to fetch Kalshi markets for {city_key} ({series}): {e}")

    # Phase 2: fetch prices concurrently with semaphore
    semaphore = asyncio.Semaphore(5)

    async def fetch_with_semaphore(ticker: str) -> tuple[float, float, float]:
        async with semaphore:
            return await _fetch_market_detail(client, ticker)

    t0 = time.monotonic()
    price_results = await asyncio.gather(*[fetch_with_semaphore(c["ticker"]) for c in candidates])
    elapsed = time.monotonic() - t0
    logger.info(f"Fetched {len(candidates)} market prices in {elapsed:.1f}s")

    # Phase 3: apply price filters and build WeatherMarket objects
    for candidate, (yes_price, no_price, volume) in zip(candidates, price_results):
        ticker = candidate["ticker"]

        # Skip resolved or illiquid markets
        if yes_price > 0.95 or yes_price < 0.05 or no_price > 0.95 or no_price < 0.05:
            logger.debug(f"Skipping {ticker} — price out of range: YES {yes_price:.0%} / NO {no_price:.0%}")
            continue

        # Only keep validated YES price sweet spot (5–30 cents)
        if yes_price > 0.30:
            logger.debug(f"Skipping {ticker} — YES {yes_price:.0%} above 30c sweet spot")
            continue

        parsed = candidate["parsed"]
        markets.append(WeatherMarket(
            slug=ticker,
            market_id=ticker,
            platform="kalshi",
            title=candidate["title"],
            city_key=candidate["city_key"],
            city_name=candidate["city_name"],
            target_date=parsed["target_date"],
            threshold_f=parsed["threshold_f"],
            metric=parsed["metric"],
            direction=parsed["direction"],
            yes_price=yes_price,
            no_price=no_price,
            volume=volume,
        ))

    logger.info(f"Found {len(markets)} Kalshi weather markets")
    return markets
