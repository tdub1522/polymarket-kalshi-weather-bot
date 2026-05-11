"""Build Markov history from 1-min candles and fetch candles from Coinbase Exchange."""
from __future__ import annotations

import math
import statistics
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


def compute_hurst(closes: List[float]) -> float:
    """R/S analysis (rescaled range) Hurst exponent.

    H > 0.55  → trending (pass filter)
    H < 0.45  → mean-reverting (block)
    0.45–0.55 → random walk (block)
    Returns 0.5 if there is insufficient data (<20 prices).
    """
    if len(closes) < 20:
        return 0.5

    n = len(closes)
    period = n // 4
    rs_values: List[float] = []

    for start in range(0, 4 * period, period):
        sub = closes[start: start + period]
        if len(sub) < 2:
            continue
        mean = sum(sub) / len(sub)
        deviations = [x - mean for x in sub]
        cumdev: List[float] = []
        running = 0.0
        for d in deviations:
            running += d
            cumdev.append(running)
        r = max(cumdev) - min(cumdev)
        try:
            s = statistics.stdev(sub)
        except statistics.StatisticsError:
            continue
        if s > 0:
            rs_values.append(r / s)

    if not rs_values:
        return 0.5

    mean_rs = sum(rs_values) / len(rs_values)
    if mean_rs <= 0 or period <= 1:
        return 0.5

    h = math.log(mean_rs) / math.log(period)
    return max(0.0, min(1.0, h))


def compute_gk_vol(candles: List[dict]) -> float:
    """Garman-Klass volatility estimator from OHLC candles.

    Returns sqrt(mean GK) — a raw vol estimate (not annualized).
    Returns 0.002 baseline if fewer than 10 candles are provided.
    """
    if len(candles) < 10:
        return 0.002

    gk_values: List[float] = []
    for c in candles:
        try:
            o = float(c["open"])
            h = float(c["high"])
            l = float(c["low"])
            cl = float(c["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if o <= 0 or h <= 0 or l <= 0 or cl <= 0:
            continue
        hl = math.log(h / l)
        co = math.log(cl / o)
        gk = 0.5 * hl ** 2 - (2 * math.log(2) - 1) * co ** 2
        if gk >= 0:
            gk_values.append(gk)

    if not gk_values:
        return 0.002

    return math.sqrt(sum(gk_values) / len(gk_values))


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
