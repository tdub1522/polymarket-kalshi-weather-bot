"""Build Markov history from 1-min candles and fetch candles from Coinbase Exchange."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import httpx

from .chain import MarkovChain, bin_state

_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


def build_history(candles: List[dict]) -> MarkovChain:
    """Populate a MarkovChain from consecutive 1-min OHLCV candles."""
    chain = MarkovChain()
    if len(candles) < 2:
        return chain

    prev_state: int | None = None
    for i in range(1, len(candles)):
        prev_close = float(candles[i - 1]["close"])
        curr_close = float(candles[i]["close"])
        if prev_close <= 0:
            continue
        pct_change = (curr_close - prev_close) / prev_close * 100.0
        state = bin_state(pct_change)
        if prev_state is not None:
            chain.add_transition(prev_state, state)
        prev_state = state

    return chain


async def fetch_1m_candles(n: int = 60) -> List[dict]:
    """Fetch the last n 1-minute candles from Coinbase Exchange REST API.

    Returns a list of dicts sorted ascending by time with keys:
    time, open, high, low, close, volume.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(seconds=(n + 3) * 60)
    params = {
        "granularity": 60,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end":   now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(_CANDLES_URL, params=params)
        r.raise_for_status()

    # Coinbase format: [[time, low, high, open, close, volume], ...] newest-first
    raw: List[list] = r.json()
    candles = [
        {
            "time":   row[0],
            "low":    row[1],
            "high":   row[2],
            "open":   row[3],
            "close":  row[4],
            "volume": row[5],
        }
        for row in raw
        if len(row) >= 6
    ]
    candles.sort(key=lambda c: c["time"])
    return candles[-n:]
