"""Discord alerter for KXBTC15M signals — separate webhook from weather.

Color scheme:
  green  — high-confidence BUY YES with good edge
  red    — high-confidence BUY NO with good edge
  blue   — low-confidence trade (manual review)
  grey   — PASS (only sent if SEND_PASS_ALERTS is on; off by default)

A short rate-limit cache prevents duplicate alerts for the same market within
the same 15-min window (Discord 429 protection).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict

import httpx

from backend.config import settings
from .agents import ExecutionSignal

logger = logging.getLogger("trading_bot")

_recent: Dict[str, datetime] = {}


def _color(signal: ExecutionSignal) -> int:
    if signal.recommendation == "PASS":
        return 0x808080
    if signal.confidence < 0.45:
        return 0x4a90e2  # blue — low confidence
    if signal.recommendation == "BUY_YES":
        return 0x00b04f  # green
    return 0xd83b3b      # red — BUY_NO


def _format_embed(signal: ExecutionSignal) -> dict:
    m = signal.market
    expiry_min = signal.market.seconds_to_expiry / 60
    edge_pct = signal.edge * 100
    yes_cents = round(m.yes_ask * 100)
    no_cents = round(m.no_ask * 100) if m.no_ask > 0 else round((1 - m.yes_bid) * 100)
    arrow = "▲" if signal.recommendation == "BUY_YES" else "▼" if signal.recommendation == "BUY_NO" else "■"

    fields = [
        {"name": "Ticker",            "value": m.market_ticker,                                     "inline": True},
        {"name": "Strike",            "value": f"${m.floor_strike:,.0f}",                            "inline": True},
        {"name": "Spot",              "value": f"${signal.spot:,.2f}",                               "inline": True},

        {"name": "Recommendation",    "value": f"**{arrow} {signal.recommendation.replace('_', ' ')}**", "inline": True},
        {"name": "Edge",              "value": f"{edge_pct:+.2f}%",                                  "inline": True},
        {"name": "Confidence",        "value": f"{signal.confidence * 100:.0f}%",                    "inline": True},

        {"name": "Model P(YES)",      "value": f"{signal.p_yes * 100:.1f}%",                         "inline": True},
        {"name": "Market P(YES)",     "value": f"{signal.market_implied_p * 100:.1f}%",              "inline": True},
        {"name": "Time to Expiry",    "value": f"{expiry_min:.1f} min",                              "inline": True},

        {"name": "YES",               "value": f"{yes_cents}¢",                                     "inline": True},
        {"name": "NO",                "value": f"{no_cents}¢",                                       "inline": True},
        {"name": "Volume",            "value": f"${m.volume:,.0f}",                                  "inline": True},
    ]

    if signal.recommendation != "PASS" and signal.contracts > 0:
        fields.append({
            "name": "Suggested Size",
            "value": f"{signal.contracts} contracts (~${signal.notional_usd:.2f}, "
                     f"Kelly={signal.kelly_fraction:.3f})",
            "inline": False,
        })

    if signal.sentiment:
        fields.append({
            "name": "Sentiment",
            "value": f"{signal.sentiment.label} ({signal.sentiment.score:+.2f}) · "
                     f"momentum: {signal.sentiment.momentum}",
            "inline": False,
        })

    if signal.thesis:
        fields.append({
            "name": "Thesis",
            "value": signal.thesis[:1000],
            "inline": False,
        })

    if signal.risk_reasons and signal.risk_reasons != ["ok"]:
        fields.append({
            "name": "Risk Notes",
            "value": ", ".join(signal.risk_reasons),
            "inline": False,
        })

    return {
        "title": f"KXBTC15M · {signal.recommendation.replace('_', ' ')}",
        "description": f"BTC vs ${m.floor_strike:,.0f} strike, {expiry_min:.0f}m to expiry",
        "color": _color(signal),
        "fields": fields,
        "footer": {"text": "Manual trade required — bot does not auto-trade"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def send_btc15m_alert(signal: ExecutionSignal) -> bool:
    """Post the signal to the BTC Discord webhook. Returns True if sent."""
    webhook = getattr(settings, "DISCORD_BTC_WEBHOOK_URL", None) or getattr(
        settings, "DISCORD_WEBHOOK_URL", None
    )
    if not webhook:
        logger.debug("KXBTC15M: no Discord webhook configured — skipping alert")
        return False

    # Skip PASS alerts unless explicitly enabled.
    if signal.recommendation == "PASS" and not getattr(settings, "KXBTC15M_SEND_PASS_ALERTS", False):
        return False

    # Rate-limit: at most one alert per market within its 15-min window.
    now = datetime.now(timezone.utc)
    key = signal.market.market_ticker
    last = _recent.get(key)
    if last and (now - last) < timedelta(minutes=14):
        logger.debug(f"KXBTC15M: rate-limited duplicate for {key}")
        return False
    _recent[key] = now
    # GC old entries.
    cutoff = now - timedelta(hours=2)
    for k in list(_recent.keys()):
        if _recent[k] < cutoff:
            del _recent[k]

    embed = _format_embed(signal)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(webhook, json={"embeds": [embed]})
            if resp.status_code == 429:
                logger.warning("KXBTC15M Discord rate-limited (429) — dropping alert")
                return False
            if resp.status_code in (200, 204):
                await asyncio.sleep(1.5)
                logger.info(f"KXBTC15M Discord alert sent: {signal.recommendation} {key}")
                return True
            logger.warning(f"KXBTC15M Discord webhook {resp.status_code}: {resp.text[:120]}")
    except Exception as exc:
        logger.error(f"KXBTC15M Discord alert failed: {exc}")
    return False
