"""MarketDiscoveryAgent — discovers the active KXBTC15M market.

Kalshi structures the 15-min Bitcoin market as one event per 15-min bucket
(US Eastern Time), with binary markets keyed by `floor_strike`. Per the
sentient-market-reader README:

  - event_ticker pattern: KXBTC15M-{YY}{MON}{DD}{HHMM}  (e.g. KXBTC15M-26MAY081730)
  - active markets satisfy yes_ask > 0
  - floor_strike is the BTC price the contract resolves above
  - close_time is the countdown anchor (NOT expiration_time)

We're robust to schema drift — fall back to series_ticker filter if the
event_ticker query returns nothing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

from backend.data.kalshi_client import KalshiClient, kalshi_credentials_present

logger = logging.getLogger("trading_bot")

SERIES_TICKER = "KXBTC15M"
ET = ZoneInfo("America/New_York")
MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


@dataclass
class Btc15mMarket:
    event_ticker: str        # e.g. KXBTC15M-26MAY081730
    market_ticker: str       # full market identifier
    title: str
    floor_strike: float      # BTC price to beat for YES resolution
    yes_bid: float           # in dollars (0..1)
    yes_ask: float
    no_bid: float
    no_ask: float
    volume: float
    close_time: datetime
    open_interest: float = 0.0

    @property
    def market_implied_p(self) -> float:
        """Treat yes_ask as the take-side YES probability."""
        if 0 < self.yes_ask < 1:
            return self.yes_ask
        if 0 < self.yes_bid < 1:
            return self.yes_bid
        return 0.5

    @property
    def seconds_to_expiry(self) -> float:
        return max(0.0, (self.close_time - datetime.now(timezone.utc)).total_seconds())


def _current_event_ticker(now_utc: Optional[datetime] = None) -> str:
    """Build the event_ticker for the currently-active 15m bucket in ET."""
    now = (now_utc or datetime.now(timezone.utc)).astimezone(ET)
    # Floor to the 15-min mark of the *current* bucket (since markets stay
    # open until close_time = bucket end).
    bucket_min = (now.minute // 15) * 15
    bucket = now.replace(minute=bucket_min, second=0, microsecond=0)
    yy = bucket.strftime("%y")
    mon = MONTH_ABBR[bucket.month - 1]
    dd = bucket.strftime("%d")
    hhmm = bucket.strftime("%H%M")
    return f"{SERIES_TICKER}-{yy}{mon}{dd}{hhmm}"


def _next_event_ticker(now_utc: Optional[datetime] = None) -> str:
    """Build the event_ticker for the next 15m bucket (used if current is closed)."""
    now = (now_utc or datetime.now(timezone.utc)).astimezone(ET)
    bucket_min = (now.minute // 15) * 15
    bucket = now.replace(minute=bucket_min, second=0, microsecond=0) + timedelta(minutes=15)
    yy = bucket.strftime("%y")
    mon = MONTH_ABBR[bucket.month - 1]
    dd = bucket.strftime("%d")
    hhmm = bucket.strftime("%H%M")
    return f"{SERIES_TICKER}-{yy}{mon}{dd}{hhmm}"


def _parse_market(raw: dict) -> Optional[Btc15mMarket]:
    try:
        floor_strike = float(raw.get("floor_strike") or raw.get("cap_strike") or 0)
        if floor_strike <= 0:
            return None

        # Kalshi sometimes returns prices as integer cents and sometimes as dollars.
        def _price(d_key: str, c_key: str) -> float:
            d = raw.get(d_key)
            if d is not None:
                return float(d)
            c = raw.get(c_key)
            if c is not None:
                return float(c) / 100.0
            return 0.0

        yes_ask = _price("yes_ask_dollars", "yes_ask")
        if yes_ask <= 0:
            return None  # filter inactive markets

        close_time_str = raw.get("close_time") or ""
        close_time = (
            datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
            if close_time_str else datetime.now(timezone.utc)
        )

        return Btc15mMarket(
            event_ticker=raw.get("event_ticker", ""),
            market_ticker=raw.get("ticker", ""),
            title=raw.get("title", ""),
            floor_strike=floor_strike,
            yes_bid=_price("yes_bid_dollars", "yes_bid"),
            yes_ask=yes_ask,
            no_bid=_price("no_bid_dollars", "no_bid"),
            no_ask=_price("no_ask_dollars", "no_ask"),
            volume=float(raw.get("volume_fp") or raw.get("volume") or 0),
            close_time=close_time,
            open_interest=float(raw.get("open_interest") or 0),
        )
    except Exception as exc:
        logger.debug(f"_parse_market failed: {exc}")
        return None


async def fetch_active_btc15m_markets(client: Optional[KalshiClient] = None) -> List[Btc15mMarket]:
    """Return the currently-active KXBTC15M markets (one event, multiple strikes).

    Strategy:
      1) Query by current event_ticker.
      2) If empty, try the next bucket (current bucket may have just closed).
      3) Fall back to series_ticker filter (broader, slower, but resilient).
    """
    if not kalshi_credentials_present():
        logger.warning("KXBTC15M: Kalshi credentials missing — skipping")
        return []

    client = client or KalshiClient()
    markets: List[Btc15mMarket] = []

    for event_ticker in (_current_event_ticker(), _next_event_ticker()):
        try:
            data = await client.get_markets({
                "event_ticker": event_ticker,
                "status": "open",
                "limit": 200,
            })
            raw_markets = data.get("markets", [])
            for raw in raw_markets:
                m = _parse_market(raw)
                if m:
                    markets.append(m)
            if markets:
                logger.info(f"KXBTC15M: discovered {len(markets)} active markets in {event_ticker}")
                return markets
        except Exception as exc:
            logger.warning(f"KXBTC15M event_ticker query failed for {event_ticker}: {exc}")

    # Fallback: series_ticker
    try:
        data = await client.get_markets({
            "series_ticker": SERIES_TICKER,
            "status": "open",
            "limit": 200,
        })
        for raw in data.get("markets", []):
            m = _parse_market(raw)
            if m:
                markets.append(m)
        logger.info(f"KXBTC15M: series fallback found {len(markets)} markets")
    except Exception as exc:
        logger.error(f"KXBTC15M series fallback failed: {exc}")

    return markets


def pick_target_market(markets: List[Btc15mMarket], spot: float) -> Optional[Btc15mMarket]:
    """Pick the most tradeable strike for the current spot.

    Heuristic: prefer near-the-money strikes (closest to spot) where yes_ask
    is in (0.05, 0.95) — outside that range the market is essentially decided.
    """
    if not markets:
        return None
    candidates = [
        m for m in markets
        if 0.05 < m.yes_ask < 0.95 and m.seconds_to_expiry > 30
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda m: abs(m.floor_strike - spot))
    return candidates[0]
