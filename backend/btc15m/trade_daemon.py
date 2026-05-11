"""Async daemon that polls the Markov pipeline around each 15-min KXBTC15M window.

Timing:
  - Wakes up 12 minutes before each :00/:15/:30/:45 ET boundary
  - Runs run_pipeline() every 60 seconds for 6 minutes (until 6 min before close)
  - The RiskManager inside the pipeline enforces the entry window; any cycle
    that falls outside the window returns PASS automatically
  - Tracks daily_trades and daily_loss in memory, resets at ET midnight
  - Sends Discord alert on BUY_YES / BUY_NO (no auto-execution)
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from loguru import logger

from backend.config import settings
from .discord_alerter import Btc15mSignal, send_btc15m_alert
from .pipeline import run_pipeline

ET = ZoneInfo("America/New_York")

_PRE_WAKEUP_MIN  = 12   # wake this many minutes before each window closes
_ENTRY_WINDOW_MIN = 6   # RiskManager blocks entries inside 6 min; we stop here
_POLL_INTERVAL    = 60  # seconds between pipeline calls inside a window


def _next_15min_boundary(now_et: datetime) -> datetime:
    """Return the next :00/:15/:30/:45 ET boundary strictly after now_et."""
    next_min = ((now_et.minute // 15) + 1) * 15
    if next_min < 60:
        return now_et.replace(minute=next_min, second=0, microsecond=0)
    return (now_et + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)


def _next_wakeup(now_et: datetime) -> datetime:
    """Return the next wakeup time: _PRE_WAKEUP_MIN minutes before the next boundary."""
    boundary = _next_15min_boundary(now_et)
    wakeup   = boundary - timedelta(minutes=_PRE_WAKEUP_MIN)
    if wakeup <= now_et:
        boundary = _next_15min_boundary(boundary + timedelta(seconds=1))
        wakeup   = boundary - timedelta(minutes=_PRE_WAKEUP_MIN)
    return wakeup


def _build_signal(result: dict) -> Btc15mSignal:
    return Btc15mSignal(
        recommendation  = result["recommendation"],
        market          = result["market"],
        spot            = result["spot"],
        p_yes           = result["p_yes"],
        edge            = result["edge"],
        market_implied_p= result["market_implied_p"],
        contracts       = result.get("contracts", 0),
        notional_usd    = result.get("notional_usd", 0.0),
        kelly_fraction  = result.get("kelly_fraction", 0.0),
        confidence      = result.get("confidence", 0.5),
        thesis          = result.get("thesis"),
        risk_reasons    = result.get("risk_reasons", []),
    )


async def _run_window(bankroll: float, daily_trades: int, daily_loss: float) -> tuple[int, float]:
    """Run the pipeline loop for one 15-min window. Returns (trades_added, loss_added)."""
    cycles          = _PRE_WAKEUP_MIN - _ENTRY_WINDOW_MIN  # 6 cycles × 60s
    trades_added    = 0
    loss_added      = 0.0

    for cycle in range(cycles):
        cycle_start = asyncio.get_event_loop().time()

        try:
            result = await run_pipeline(
                bankroll      = bankroll,
                daily_trades  = daily_trades + trades_added,
                daily_loss    = daily_loss + loss_added,
            )
        except Exception as exc:
            logger.error(f"Pipeline exception: {exc}")
            result = {"recommendation": "PASS", "reason": str(exc)}

        rec    = result.get("recommendation", "PASS")
        reason = result.get("reason", "")

        logger.info(
            "Cycle {}/{}: {} {}",
            cycle + 1, cycles, rec,
            f"| {reason}" if reason else f"| p_yes={result.get('p_yes', '?'):.3f}"
            if "p_yes" in result else "",
        )

        if rec in ("BUY_YES", "BUY_NO"):
            try:
                signal = _build_signal(result)
                sent   = await send_btc15m_alert(signal)
                if sent:
                    trades_added += 1
                    cost = (
                        result["market"].yes_ask
                        if rec == "BUY_YES"
                        else result["market"].no_ask
                    )
                    loss_added += result.get("contracts", 0) * cost
                    logger.info(
                        "Alert sent: {} {} contracts @ {:.2f} (notional ${:.2f})",
                        rec,
                        result.get("contracts", 0),
                        cost,
                        result.get("notional_usd", 0.0),
                    )
            except Exception as exc:
                logger.error(f"Discord alert failed: {exc}")

        elapsed = asyncio.get_event_loop().time() - cycle_start
        await asyncio.sleep(max(0.0, _POLL_INTERVAL - elapsed))

    return trades_added, loss_added


async def run_daemon(bankroll: float = 80.0) -> None:
    """Run forever, waking around each 15-min window. Ctrl-C to stop."""
    pem_contents = os.getenv("KALSHI_PRIVATE_KEY_CONTENTS")
    pem_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "/app/kalshi-prod.pem")
    if pem_contents and not os.path.exists(pem_path):
        os.makedirs(os.path.dirname(pem_path), exist_ok=True)
        with open(pem_path, "w") as f:
            f.write(pem_contents)

    daily_trades    = 0
    daily_loss      = 0.0
    last_reset_date = datetime.now(ET).date()

    logger.info("KXBTC15M trade daemon starting (bankroll=${:.2f})", bankroll)

    while True:
        now_et = datetime.now(ET)

        # Midnight ET reset
        if now_et.date() > last_reset_date:
            logger.info(
                "Daily reset: trades={} loss=${:.2f} → 0",
                daily_trades, daily_loss,
            )
            daily_trades    = 0
            daily_loss      = 0.0
            last_reset_date = now_et.date()

        wakeup      = _next_wakeup(now_et)
        sleep_secs  = (wakeup - datetime.now(ET)).total_seconds()
        logger.info(
            "Sleeping {:.0f}s → wakeup at {} ET",
            sleep_secs,
            wakeup.strftime("%H:%M:%S"),
        )
        await asyncio.sleep(max(1.0, sleep_secs))

        logger.info("Window open — running {} pipeline cycles", _PRE_WAKEUP_MIN - _ENTRY_WINDOW_MIN)
        try:
            added_trades, added_loss = await _run_window(bankroll, daily_trades, daily_loss)
            daily_trades += added_trades
            daily_loss   += added_loss
        except Exception as exc:
            logger.error(f"Window loop crashed: {exc}")

        logger.info(
            "Window closed. daily_trades={} daily_loss=${:.2f}",
            daily_trades, daily_loss,
        )


if __name__ == "__main__":
    try:
        asyncio.run(run_daemon(bankroll=getattr(settings, "INITIAL_BANKROLL", 80.0)))
    except KeyboardInterrupt:
        logger.info("Daemon stopped by user")
