"""Discord webhook notifications for trade signals."""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import logging
logger = logging.getLogger("trading_bot")

from backend.config import settings

_recently_sent: dict[str, datetime] = {}


async def send_discord_signal(signal: dict[str, Any]) -> None:
    global _recently_sent

    if not settings.DISCORD_ENABLED or not settings.DISCORD_WEBHOOK_URL:
        logger.debug("Discord disabled or webhook not configured — skipping notification")
        return

    ticker = signal.get("ticker", "")
    now = datetime.now(timezone.utc)

    if ticker in _recently_sent:
        time_since_last = now - _recently_sent[ticker]
        if time_since_last < timedelta(hours=1):
            logger.debug(f"Skipping Discord alert for {ticker} — sent {time_since_last.seconds // 60}m ago")
            return

    _recently_sent[ticker] = now
    _recently_sent = {k: v for k, v in _recently_sent.items() if now - v < timedelta(hours=2)}

    yes_cents = round(signal.get('yes_price', 0) * 100)
    no_cents = round(signal.get('no_price', 0) * 100)
    members = signal.get('ensemble_members', 'N/A')
    ensemble_mean = signal.get('ensemble_mean', 0)
    ensemble_std = signal.get('ensemble_std', 0)

    yes_price = signal.get("yes_price", 0)
    market_meta = signal.get("market", {})
    threshold_f = market_meta.get("threshold_f", 0)
    market_direction = market_meta.get("direction", "below")

    if market_direction == "above":
        mt_distance_val = threshold_f - ensemble_mean
        mt_distance_label = f"{mt_distance_val:.1f}°F below threshold"
    else:
        mt_distance_val = ensemble_mean - threshold_f
        mt_distance_label = f"{mt_distance_val:.1f}°F above threshold"

    if yes_price <= 0.10:
        color = 0x00ff00   # green — best signals, highest historical win rate
    elif yes_price <= 0.20:
        color = 0xffaa00   # yellow
    else:
        color = 0xff6600   # orange

    fields = [
        {"name": "Ticker",               "value": signal.get("ticker", "N/A"),                                                    "inline": True},
        {"name": "Side",                 "value": signal.get("side", "N/A").upper(),                                              "inline": True},
        {"name": "YES Price",            "value": f"{yes_cents}¢",                                                               "inline": True},
        {"name": "NO Price",             "value": f"{no_cents}¢",                                                                  "inline": True},

        {"name": "Expected Value",       "value": f"{signal.get('expected_value', 0) * 100:.1f}%",                                "inline": True},
        {"name": "Historical Win Rate",  "value": f"{signal.get('hist_win_rate', 0) * 100:.1f}%",                                 "inline": True},
        {"name": "Position Size",        "value": f"${signal.get('suggested_size', 0):.0f} (confidence: {signal.get('confidence', 0)*100:.0f}%)", "inline": True},
        {"name": f"MT Oracle Mean ({members} members)", "value": f"{ensemble_mean:.1f}°F",                                       "inline": True},
        {"name": "MT Oracle Std",        "value": f"±{ensemble_std:.1f}°F",                                                       "inline": True},
        {"name": "MT Oracle Distance",   "value": mt_distance_label,                                                               "inline": True},
    ]

    embed = {
        "title": signal.get("market_title", "Unknown Market"),
        "color": color,
        "fields": fields,
        "footer": {
            "text": "Manual trade required — bot does not auto-trade"
        },
    }

    payload = {"embeds": [embed]}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(settings.DISCORD_WEBHOOK_URL, json=payload)
            if resp.status_code == 429:
                logger.warning("Discord rate limited (429) — skipping this signal")
                return
            if resp.status_code in (200, 204):
                await asyncio.sleep(1.5)
            else:
                logger.warning("Discord webhook returned {}: {}", resp.status_code, resp.text[:100])
    except Exception as exc:
        logger.error("Discord notification failed: {}", exc)
