"""PriceFeedAgent — BTC spot via Coinbase + Kraken median (BRTI proxy).

Kalshi KXBTC15M settles to CF Benchmarks BRTI, which is calculated from a
basket of regulated exchanges (Coinbase, Kraken, Bitstamp, LMAX Digital,
itBit). Coinbase + Kraken are the two highest-volume free-API constituents,
so a simple median of the two is a much better proxy for BRTI than a single
quote from CoinMarketCap (which is a derived aggregate, not a constituent).

Both APIs are public (no key) and rate-limit forgiving for our 5-min cadence.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Deque, List, Optional, Tuple

import httpx

logger = logging.getLogger("trading_bot")

COINBASE_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
KRAKEN_URL = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
TIMEOUT = 5.0


@dataclass
class BtcPriceSnapshot:
    ts: float
    spot: float           # median of constituent exchanges
    coinbase: Optional[float]
    kraken: Optional[float]
    sources: List[str] = field(default_factory=list)


class BtcPriceFeed:
    """Rolling BTC price history with a BRTI-style median across constituents."""

    def __init__(self, history_seconds: int = 3600):
        self.history_seconds = history_seconds
        self._history: Deque[BtcPriceSnapshot] = deque()

    async def fetch(self) -> Optional[BtcPriceSnapshot]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            cb_task = asyncio.create_task(_fetch_coinbase(client))
            kr_task = asyncio.create_task(_fetch_kraken(client))
            cb, kr = await asyncio.gather(cb_task, kr_task, return_exceptions=True)

        cb_price = cb if isinstance(cb, (int, float)) else None
        kr_price = kr if isinstance(kr, (int, float)) else None
        prices = [p for p in (cb_price, kr_price) if p is not None and p > 0]
        if not prices:
            logger.warning("BTC price feed: all constituents failed")
            return None

        sources = []
        if cb_price:
            sources.append("coinbase")
        if kr_price:
            sources.append("kraken")
        snap = BtcPriceSnapshot(
            ts=time.time(),
            spot=median(prices),
            coinbase=cb_price,
            kraken=kr_price,
            sources=sources,
        )
        self._record(snap)
        return snap

    def _record(self, snap: BtcPriceSnapshot) -> None:
        self._history.append(snap)
        cutoff = time.time() - self.history_seconds
        while self._history and self._history[0].ts < cutoff:
            self._history.popleft()

    @property
    def history(self) -> List[BtcPriceSnapshot]:
        return list(self._history)

    def change_over(self, seconds: int) -> Optional[float]:
        """Return percent change in spot over the last `seconds`."""
        if not self._history:
            return None
        now = time.time()
        target = now - seconds
        latest = self._history[-1].spot
        # Find oldest snapshot within window.
        anchor: Optional[BtcPriceSnapshot] = None
        for snap in self._history:
            if snap.ts >= target:
                anchor = snap
                break
        if anchor is None or anchor.spot <= 0:
            return None
        return (latest - anchor.spot) / anchor.spot

    def realized_vol(self, seconds: int = 900) -> Optional[float]:
        """Approximate realized vol over the window as stdev of log returns."""
        import math
        cutoff = time.time() - seconds
        pts = [s.spot for s in self._history if s.ts >= cutoff]
        if len(pts) < 3:
            return None
        rets = [math.log(pts[i] / pts[i - 1]) for i in range(1, len(pts)) if pts[i - 1] > 0]
        if len(rets) < 2:
            return None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var)


async def _fetch_coinbase(client: httpx.AsyncClient) -> Optional[float]:
    try:
        r = await client.get(COINBASE_URL)
        r.raise_for_status()
        amt = r.json().get("data", {}).get("amount")
        return float(amt) if amt else None
    except Exception as exc:
        logger.debug(f"Coinbase fetch failed: {exc}")
        return None


async def _fetch_kraken(client: httpx.AsyncClient) -> Optional[float]:
    try:
        r = await client.get(KRAKEN_URL)
        r.raise_for_status()
        data = r.json().get("result", {})
        # Kraken returns one key like "XXBTZUSD"; grab whichever it is.
        for _, v in data.items():
            last = v.get("c", [None])[0]
            if last:
                return float(last)
    except Exception as exc:
        logger.debug(f"Kraken fetch failed: {exc}")
    return None


# Module-level singleton — keeps a rolling history across pipeline cycles.
_feed_singleton: Optional[BtcPriceFeed] = None


def get_feed() -> BtcPriceFeed:
    global _feed_singleton
    if _feed_singleton is None:
        _feed_singleton = BtcPriceFeed()
    return _feed_singleton
