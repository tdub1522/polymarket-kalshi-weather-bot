"""Stages 3-6 of the pipeline: Sentiment, ProbabilityModel, RiskManager, Execution.

Stages 1-2 (MarketDiscovery, PriceFeed) live in their own modules because they
have no LLM dependency and are independently useful.

The Risk and Execution stages are intentionally NOT LLM-powered — safety-critical
logic must be deterministic and auditable. Same design choice as the upstream repo.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .llm import Llm
from .roma import solve, RomaResult
from .kalshi_market import Btc15mMarket
from .price_feed import BtcPriceSnapshot

logger = logging.getLogger("trading_bot")


# ─────────────────── Stage 3: Sentiment ───────────────────

_SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},          # [-1, +1]
        "label": {"type": "string"},          # "bullish" | "neutral" | "bearish"
        "momentum": {"type": "string"},       # "up" | "flat" | "down"
        "signals": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "label", "momentum"],
}


@dataclass
class SentimentResult:
    score: float
    label: str
    momentum: str
    signals: List[str] = field(default_factory=list)


async def run_sentiment(llm: Llm, ctx: Dict) -> SentimentResult:
    """Fast-tier directional sentiment over recent BTC behavior."""
    prompt = (
        "Score the very-short-term (next 15 minutes) directional sentiment for BTC "
        "based on the data below. Return JSON with score in [-1,+1] (negative=bearish, "
        "positive=bullish), label, momentum, and 1-3 signal strings.\n\n"
        f"Spot: {ctx.get('spot')}\n"
        f"15m change: {ctx.get('spot_15m_change')}\n"
        f"1h change: {ctx.get('spot_1h_change')}\n"
        f"Realized vol (15m): {ctx.get('realized_vol_15m')}\n"
        f"Distance to strike (bps): {ctx.get('distance_to_strike_bps')}\n"
        f"Seconds to expiry: {ctx.get('seconds_to_expiry')}\n"
    )
    res = await llm.tool_call_json(prompt, schema=_SENTIMENT_SCHEMA, tier="fast", max_tokens=300)
    if res.parsed_json:
        return SentimentResult(
            score=max(-1.0, min(1.0, float(res.parsed_json.get("score", 0)))),
            label=str(res.parsed_json.get("label", "neutral"))[:16],
            momentum=str(res.parsed_json.get("momentum", "flat"))[:8],
            signals=list(res.parsed_json.get("signals", []))[:3],
        )
    # Heuristic fallback: use 15m + 1h change.
    ch_15 = ctx.get("spot_15m_change") or 0.0
    ch_1h = ctx.get("spot_1h_change") or 0.0
    score = max(-1.0, min(1.0, ch_15 * 50 + ch_1h * 10))
    return SentimentResult(
        score=score,
        label="bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral",
        momentum="up" if ch_15 > 0 else "down" if ch_15 < 0 else "flat",
        signals=[f"15m={ch_15:+.4f}", f"1h={ch_1h:+.4f}"],
    )


# ─────────────────── Stage 4: ProbabilityModel (ROMA) ───────────────────


async def run_probability_model(llm: Llm, ctx: Dict) -> RomaResult:
    """Run ROMA recursive solve over the trading goal."""
    goal = (
        f"Decide whether to BUY YES, BUY NO, or PASS on Kalshi market {ctx.get('ticker')}. "
        f"Contract resolves YES if BTC closes above {ctx.get('strike')} at expiry. "
        f"Produce a calibrated P(YES), a directional recommendation, and a confidence."
    )
    return await solve(goal, ctx, llm)


# ─────────────────── Stage 5: RiskManager (deterministic) ───────────────────


@dataclass
class RiskDecision:
    approved: bool
    contracts: int
    notional_usd: float
    kelly_fraction: float
    reasons: List[str] = field(default_factory=list)


def run_risk_manager(
    *,
    p_yes: float,
    market: Btc15mMarket,
    recommendation: str,
    bankroll: float,
    daily_pnl: float,
    daily_loss_cap: float,
    drawdown_pct: float,
    max_drawdown_pct: float,
    trades_today: int,
    max_trades_per_day: int,
    max_trade_size: float,
    kelly_fraction_cap: float = 0.25,
    min_edge: float = 0.03,
) -> RiskDecision:
    """Quarter-Kelly sizing with hard caps. Mirrors the repo's risk rules but
    uses Trey's existing conservative defaults."""
    reasons: List[str] = []
    if recommendation == "PASS":
        return RiskDecision(False, 0, 0.0, 0.0, ["recommendation_pass"])

    # Compute side-adjusted edge.
    market_p = market.market_implied_p
    if recommendation == "BUY_YES":
        # Pay yes_ask, win 1.0 if YES.
        cost = market.yes_ask
        win_prob = p_yes
    else:
        # Pay no_ask, win 1.0 if NO.
        cost = market.no_ask if market.no_ask > 0 else (1 - market.yes_bid)
        win_prob = 1.0 - p_yes

    if cost <= 0 or cost >= 1:
        return RiskDecision(False, 0, 0.0, 0.0, ["bad_cost"])

    edge = win_prob - cost
    if edge < min_edge:
        return RiskDecision(False, 0, 0.0, 0.0, [f"edge<{min_edge}"])

    # Daily loss / drawdown / trade-count gates.
    if daily_pnl <= -daily_loss_cap:
        reasons.append("daily_loss_cap_hit")
    if drawdown_pct >= max_drawdown_pct:
        reasons.append("drawdown_cap_hit")
    if trades_today >= max_trades_per_day:
        reasons.append("trade_count_cap_hit")
    if reasons:
        return RiskDecision(False, 0, 0.0, 0.0, reasons)

    # Kelly fraction for binary contracts: f* = (p*b - q) / b
    # where b = (1-cost)/cost, q = 1-p. Then scale by kelly_fraction_cap.
    b = (1 - cost) / cost
    q = 1 - win_prob
    full_kelly = max(0.0, (win_prob * b - q) / b) if b > 0 else 0.0
    kelly = full_kelly * kelly_fraction_cap

    # Notional dollars to risk = kelly * bankroll, capped.
    risk_notional = min(bankroll * kelly, max_trade_size)
    if risk_notional <= 0:
        return RiskDecision(False, 0, 0.0, full_kelly, ["kelly_nonpositive"])

    # contracts = floor(risk_notional / cost). Each contract risks `cost` dollars.
    contracts = max(1, int(risk_notional / cost))

    return RiskDecision(
        approved=True,
        contracts=contracts,
        notional_usd=contracts * cost,
        kelly_fraction=full_kelly,
        reasons=["ok"],
    )


# ─────────────────── Stage 6: Execution (deterministic, signal-only) ───────────────────


@dataclass
class ExecutionSignal:
    """The final pipeline output — what gets posted to Discord."""
    market: Btc15mMarket
    spot: float
    p_yes: float
    market_implied_p: float
    edge: float                  # signed: positive = lean YES
    recommendation: str          # BUY_YES | BUY_NO | PASS
    contracts: int
    notional_usd: float
    kelly_fraction: float
    confidence: float
    sentiment: SentimentResult
    thesis: str
    key_drivers: List[str] = field(default_factory=list)
    risk_reasons: List[str] = field(default_factory=list)
    auto_executed: bool = False
    error: Optional[str] = None


def build_execution_signal(
    *,
    market: Btc15mMarket,
    spot: float,
    sentiment: SentimentResult,
    roma: RomaResult,
    risk: RiskDecision,
) -> ExecutionSignal:
    edge = roma.p_yes - market.market_implied_p
    return ExecutionSignal(
        market=market,
        spot=spot,
        p_yes=roma.p_yes,
        market_implied_p=market.market_implied_p,
        edge=edge,
        recommendation=roma.recommendation if risk.approved else "PASS",
        contracts=risk.contracts,
        notional_usd=risk.notional_usd,
        kelly_fraction=risk.kelly_fraction,
        confidence=roma.confidence,
        sentiment=sentiment,
        thesis=roma.thesis,
        key_drivers=roma.key_drivers,
        risk_reasons=risk.reasons,
    )
