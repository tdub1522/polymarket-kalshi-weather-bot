"""Aggregator — fast-tier synthesizer.

Takes the list of (subtask, answer) pairs and produces:
  - a unified market thesis (string)
  - a single calibrated P(YES) estimate
  - a directional recommendation: BUY_YES | BUY_NO | PASS
  - a confidence in [0,1]

The structured extraction is the gateway to deterministic risk sizing.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple
from ..llm import Llm

_AGG_SCHEMA = {
    "type": "object",
    "properties": {
        "thesis": {"type": "string"},
        "p_yes": {"type": "number"},
        "recommendation": {"type": "string", "enum": ["BUY_YES", "BUY_NO", "PASS"]},
        "confidence": {"type": "number"},
        "key_drivers": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["thesis", "p_yes", "recommendation", "confidence"],
}


async def synthesize(
    llm: Llm,
    goal: str,
    subtask_results: List[Tuple[str, str]],
    context: Dict,
) -> Dict:
    """Aggregate executor outputs into a single structured trading view."""
    sections = "\n\n".join(
        f"### Subtask {i+1}: {q}\nAnswer: {a}" for i, (q, a) in enumerate(subtask_results)
    )
    market_implied = context.get("market_implied_p", 0.5)
    seconds_left = context.get("seconds_to_expiry", 0)
    spot = context.get("spot", 0)
    strike = context.get("strike", 0)

    prompt = (
        f"You are aggregating a multi-agent analysis of a Kalshi KXBTC15M binary market.\n"
        f"The contract resolves YES if BTC closes above {strike:.2f} at expiry "
        f"({seconds_left:.0f}s away). Current spot: {spot:.2f}. "
        f"Market-implied P(YES) (yes_ask): {market_implied:.3f}.\n\n"
        f"Goal: {goal}\n\nSub-analyses:\n{sections}\n\n"
        "Synthesize into a calibrated probability and directional view. "
        "Recommendation rules: PASS unless |p_yes - market_implied_p| >= 0.03 "
        "(3% min edge). BUY_YES if p_yes > market_implied + 0.03. BUY_NO if "
        "p_yes < market_implied - 0.03. Return JSON only."
    )
    res = await llm.tool_call_json(prompt, schema=_AGG_SCHEMA, tier="fast", max_tokens=800)
    if res.parsed_json:
        return _validate(res.parsed_json, market_implied)
    # Fallback — pure orderbook/momentum heuristic when LLM is unavailable.
    return _heuristic_fallback(context)


def _validate(parsed: Dict, market_implied: float) -> Dict:
    """Clamp and sanity-check the aggregator's output."""
    p = float(parsed.get("p_yes", market_implied))
    p = max(0.005, min(0.995, p))
    rec = parsed.get("recommendation", "PASS")
    if rec not in ("BUY_YES", "BUY_NO", "PASS"):
        rec = "PASS"

    edge = p - market_implied
    # Re-derive recommendation deterministically — never trust the LLM with
    # the trade direction logic.
    if edge >= 0.03:
        rec = "BUY_YES"
    elif edge <= -0.03:
        rec = "BUY_NO"
    else:
        rec = "PASS"

    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    return {
        "thesis": str(parsed.get("thesis", "")).strip()[:1200],
        "p_yes": p,
        "recommendation": rec,
        "confidence": confidence,
        "key_drivers": parsed.get("key_drivers", [])[:6],
    }


def _heuristic_fallback(context: Dict) -> Dict:
    """No-LLM fallback — same structure, deterministic logic.

    Uses spot vs strike, time remaining, and realized vol to produce a
    Black-Scholes-flavored P(YES) estimate; clamped and routed through the
    same recommendation gate.
    """
    import math

    def _f(key, default=0.0):
        v = context.get(key)
        try:
            return float(v) if v is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    spot = _f("spot")
    strike = _f("strike")
    sec = max(1.0, _f("seconds_to_expiry"))
    market_p = _f("market_implied_p", 0.5)

    # Annualized BTC vol estimate from realized 15m vol; default 60% annual if missing.
    rv = _f("realized_vol_15m")
    sigma_ann = max(0.20, min(2.50, rv * math.sqrt(365 * 24 * 4))) if rv > 0 else 0.60
    t_years = sec / (365 * 24 * 3600)

    if spot <= 0 or strike <= 0 or t_years <= 0:
        p = market_p
    else:
        # P(S_T > K) under GBM with zero drift: N(d2) where d2 = ln(S/K)/(sigma*sqrt(T))
        d2 = math.log(spot / strike) / (sigma_ann * math.sqrt(t_years))
        p = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2)))

    p = max(0.005, min(0.995, p))
    edge = p - market_p
    if edge >= 0.03:
        rec = "BUY_YES"
    elif edge <= -0.03:
        rec = "BUY_NO"
    else:
        rec = "PASS"
    return {
        "thesis": (
            f"[fallback] GBM model: spot {spot:.2f}, strike {strike:.2f}, "
            f"{sec:.0f}s to expiry, sigma={sigma_ann:.2f} → P(YES)={p:.3f} "
            f"vs market {market_p:.3f}. Edge {edge*100:+.2f}%."
        ),
        "p_yes": p,
        "recommendation": rec,
        "confidence": 0.4,  # low confidence — no LLM input
        "key_drivers": ["heuristic_gbm"],
    }
