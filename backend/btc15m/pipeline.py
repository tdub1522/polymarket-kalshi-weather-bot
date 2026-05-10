"""6-stage KXBTC15M pipeline orchestrator.

   Stage 1  MarketDiscoveryAgent  -> kalshi_market.fetch_active_btc15m_markets
   Stage 2  PriceFeedAgent        -> price_feed.BtcPriceFeed
   Stage 3  SentimentAgent        -> agents.run_sentiment              [LLM fast]
   Stage 4  ProbabilityModelAgent -> agents.run_probability_model      [ROMA]
   Stage 5  RiskManagerAgent      -> agents.run_risk_manager           [deterministic]
   Stage 6  ExecutionAgent        -> agents.build_execution_signal     [deterministic]

The pipeline is signal-only by design. Even if `TRADING_ENABLED` is set we do
NOT call any order placement endpoint from this module — that's a separate,
gated path the user adds when they're ready.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from backend.config import settings

_ET = ZoneInfo("America/New_York")


def _within_active_hours() -> bool:
    """Return True if the current ET time is within allowed trading hours.

    Blocked windows:
      - Saturday 00:00 ET through Sunday 18:00 ET (weekend market closure)
      - Any other day outside 06:00–midnight ET
    """
    now_et = datetime.now(_ET)
    weekday = now_et.weekday()   # 0=Mon … 5=Sat, 6=Sun
    hour    = now_et.hour

    if weekday == 5:             # Saturday — always blocked
        return False
    if weekday == 6:             # Sunday — blocked before 18:00
        return hour >= 18
    # Mon–Fri: allowed 06:00–23:59
    return hour >= 6

from .agents import (
    SentimentResult,
    build_execution_signal,
    run_probability_model,
    run_risk_manager,
    run_sentiment,
)
from .agents import ExecutionSignal
from .kalshi_market import Btc15mMarket, fetch_active_btc15m_markets, pick_target_market
from .llm import Llm
from .price_feed import BtcPriceSnapshot, get_feed

logger = logging.getLogger("trading_bot")


@dataclass
class PipelineRunResult:
    signal: Optional[ExecutionSignal]
    markets_seen: int
    elapsed_ms: float
    error: Optional[str] = None


async def run_btc15m_pipeline(
    *,
    bankroll: float,
    daily_pnl: float = 0.0,
    drawdown_pct: float = 0.0,
    trades_today: int = 0,
) -> PipelineRunResult:
    """Run one full pipeline cycle. Returns the resulting signal (or None)."""
    if not _within_active_hours():
        logger.info("Scan skipped — outside active hours")
        return PipelineRunResult(None, 0, 0.0)

    started = time.monotonic()

    # Stage 1 — Market discovery.
    try:
        markets = await fetch_active_btc15m_markets()
    except Exception as exc:
        logger.exception("KXBTC15M discovery failed")
        return PipelineRunResult(None, 0, (time.monotonic() - started) * 1000, str(exc))

    # Stage 2 — Price feed (BRTI proxy).
    feed = get_feed()
    snap = await feed.fetch()
    if not snap:
        return PipelineRunResult(None, len(markets), (time.monotonic() - started) * 1000, "no_price")

    target = pick_target_market(markets, snap.spot)
    if target is None:
        logger.info("KXBTC15M: no tradeable strike near spot — skipping cycle")
        return PipelineRunResult(None, len(markets), (time.monotonic() - started) * 1000)

    # Build pipeline context — passed to every LLM stage.
    distance_to_strike = target.floor_strike - snap.spot
    distance_to_strike_bps = (
        10000.0 * distance_to_strike / snap.spot if snap.spot > 0 else 0.0
    )
    ctx = {
        "ticker": target.market_ticker,
        "event_ticker": target.event_ticker,
        "strike": target.floor_strike,
        "yes_bid": target.yes_bid,
        "yes_ask": target.yes_ask,
        "no_bid": target.no_bid,
        "no_ask": target.no_ask,
        "volume": target.volume,
        "spot": snap.spot,
        "spot_15m_change": feed.change_over(900),
        "spot_1h_change": feed.change_over(3600),
        "realized_vol_15m": feed.realized_vol(900),
        "seconds_to_expiry": target.seconds_to_expiry,
        "distance_to_strike": distance_to_strike,
        "distance_to_strike_bps": distance_to_strike_bps,
        "market_implied_p": target.market_implied_p,
        "price_history": [
            {"ts": s.ts, "p": s.spot} for s in feed.history[-12:]
        ],
    }

    llm = Llm()

    # Stage 3 — Sentiment.
    sentiment = await run_sentiment(llm, ctx)

    # Stage 4 — ROMA probability model.
    roma = await run_probability_model(llm, ctx)
    logger.info(
        f"ROMA → P(YES)={roma.p_yes:.3f} rec={roma.recommendation} "
        f"conf={roma.confidence:.2f} ({roma.elapsed_ms:.0f}ms, {len(roma.subtasks)} subtasks)"
    )

    # Stage 5 — Risk.
    risk = run_risk_manager(
        p_yes=roma.p_yes,
        market=target,
        recommendation=roma.recommendation,
        bankroll=bankroll,
        daily_pnl=daily_pnl,
        daily_loss_cap=getattr(settings, "KXBTC15M_DAILY_LOSS_CAP", settings.DAILY_LOSS_LIMIT),
        drawdown_pct=drawdown_pct,
        max_drawdown_pct=getattr(settings, "KXBTC15M_MAX_DRAWDOWN_PCT", 0.15),
        trades_today=trades_today,
        max_trades_per_day=getattr(settings, "KXBTC15M_MAX_TRADES_PER_DAY", 24),
        max_trade_size=getattr(settings, "KXBTC15M_MAX_TRADE_SIZE", settings.MAX_TRADE_SIZE),
        kelly_fraction_cap=settings.KELLY_FRACTION,
        min_edge=getattr(settings, "KXBTC15M_MIN_EDGE", 0.03),
    )

    # Stage 6 — Execution (signal generation only).
    signal = build_execution_signal(
        market=target,
        spot=snap.spot,
        sentiment=sentiment,
        roma=roma,
        risk=risk,
    )
    return PipelineRunResult(signal, len(markets), (time.monotonic() - started) * 1000)
