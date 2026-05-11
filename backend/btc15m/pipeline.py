"""KXBTC15M Markov-chain pipeline — signal-only, no order placement."""
from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

from .kalshi_market import fetch_active_btc15m_markets, pick_target_market
from .markov.chain import STATES
from .markov.history import build_history, fetch_1m_candles
from .risk.manager import RiskManager

_PASS = "PASS"
_ET   = ZoneInfo("America/New_York")
logger = logging.getLogger("trading_bot")


def _within_active_hours() -> bool:
    """Return True if the current ET time is within allowed trading hours.

    Blocked windows:
      - Saturday 00:00 ET through Sunday 18:00 ET (weekend market closure)
      - Any day outside 07:00–21:59 ET
    """
    now     = datetime.now(_ET)
    weekday = now.weekday()   # 0=Mon … 5=Sat, 6=Sun
    hour    = now.hour

    if weekday == 5:          # Saturday — always blocked
        return False
    if weekday == 6:          # Sunday — blocked before 18:00 ET
        if hour < 18:
            return False
    return 7 <= hour < 22


def _sigma_approx(candles: list, spot: float, steps: int = 15) -> float:
    """Estimate price-level sigma over `steps` 1-min bars from candle history."""
    closes = [float(c["close"]) for c in candles if c.get("close")]
    if len(closes) < 3:
        return spot * 0.001 * math.sqrt(steps)
    log_rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    if len(log_rets) < 2:
        return spot * 0.001 * math.sqrt(steps)
    stdev = statistics.stdev(log_rets)
    return spot * stdev * math.sqrt(steps)


async def run_pipeline(
    bankroll: float = 80.0,
    daily_trades: int = 0,
    daily_loss: float = 0.0,
) -> Dict[str, Any]:
    """Run one full Markov-chain pipeline cycle. Returns a signal dict."""

    # ── 0. Active-hours gate ─────────────────────────────────────────────────
    if not _within_active_hours():
        logger.info("Scan skipped — outside active hours")
        return {"recommendation": _PASS, "reason": "outside active hours"}

    # ── 1. Candles ───────────────────────────────────────────────────────────
    try:
        candles = await fetch_1m_candles(60)
    except Exception as exc:
        return {"recommendation": _PASS, "reason": f"candle fetch failed: {exc}"}

    if len(candles) < 2:
        return {"recommendation": _PASS, "reason": "too few candles"}

    spot = float(candles[-1]["close"])

    # ── 2. Build Markov chain ─────────────────────────────────────────────────
    chain = build_history(candles)

    # ── 3. Validity gate ──────────────────────────────────────────────────────
    if not chain.is_valid():
        return {"recommendation": _PASS, "reason": "insufficient Markov history"}

    # ── 4. Persistence gate ───────────────────────────────────────────────────
    dominant_state, persistence = chain.get_dominant_state()
    if persistence < 0.82:
        return {
            "recommendation": _PASS,
            "reason": f"persistence {persistence:.3f} < 0.82",
        }

    # ── 5. Propagate & p_yes ──────────────────────────────────────────────────
    chain.propagate(15)
    sigma = _sigma_approx(candles, spot, steps=15)

    # ── 6. Gap gate ───────────────────────────────────────────────────────────
    # We don't have the strike yet, but the gap check is on p_yes vs 0.50.
    # Use a placeholder strike equal to spot so p_yes ~ directional strength.
    p_yes_raw = chain.p_yes(spot, spot, sigma)
    gap = abs(p_yes_raw - 0.50)
    if gap < 0.11:
        return {
            "recommendation": _PASS,
            "reason": f"gap {gap:.3f} < 0.11",
        }

    # ── 7. Market discovery ───────────────────────────────────────────────────
    try:
        markets = await fetch_active_btc15m_markets()
    except Exception as exc:
        return {"recommendation": _PASS, "reason": f"market fetch failed: {exc}"}

    target = pick_target_market(markets, spot)
    if target is None:
        return {"recommendation": _PASS, "reason": "no tradeable market near spot"}

    # Recompute p_yes against the actual strike.
    p_yes = chain.p_yes(target.floor_strike, spot, sigma)

    # ── 8/9. Recommendation ───────────────────────────────────────────────────
    if p_yes > 0.61:
        recommendation = "BUY_YES"
    elif p_yes < 0.39:
        recommendation = "BUY_NO"
    else:
        return {
            "recommendation": _PASS,
            "reason": f"p_yes {p_yes:.3f} in neutral zone [0.39, 0.61]",
        }

    # ── 10. Risk gates ────────────────────────────────────────────────────────
    risk = RiskManager(daily_trades=daily_trades, daily_loss=daily_loss)
    passed, risk_reasons = risk.check_all_gates(
        market=target,
        spot=spot,
        recommendation=recommendation,
        yes_ask=target.yes_ask,
        no_ask=target.no_ask,
        bankroll=bankroll,
    )
    if not passed:
        return {
            "recommendation": _PASS,
            "reason": "risk gate: " + "; ".join(risk_reasons),
            "risk_reasons": risk_reasons,
        }

    # ── 11. Kelly sizing ──────────────────────────────────────────────────────
    contracts = risk.kelly_contracts(
        p_yes=p_yes,
        yes_ask=target.yes_ask,
        no_ask=target.no_ask,
        recommendation=recommendation,
        bankroll=bankroll,
    )
    cost_per_ctr  = target.yes_ask if recommendation == "BUY_YES" else target.no_ask
    notional_usd  = contracts * cost_per_ctr
    kelly_fraction = notional_usd / bankroll if bankroll > 0 else 0.0

    edge = (
        p_yes - target.yes_ask
        if recommendation == "BUY_YES"
        else (1.0 - p_yes) - target.no_ask
    )

    thesis = (
        f"Markov dominant={STATES[dominant_state]} persistence={persistence:.3f} "
        f"p_yes={p_yes:.3f} gap={gap:.3f} edge={edge:+.3f}"
    )

    # ── 12. Signal dict ───────────────────────────────────────────────────────
    return {
        "recommendation":  recommendation,
        "p_yes":           p_yes,
        "market":          target,
        "spot":            spot,
        "edge":            edge,
        "market_implied_p": target.market_implied_p,
        "contracts":       contracts,
        "notional_usd":    notional_usd,
        "kelly_fraction":  kelly_fraction,
        "confidence":      persistence,
        "thesis":          thesis,
        "risk_reasons":    [],
        "dominant_state":  STATES[dominant_state],
        "persistence":     persistence,
    }
