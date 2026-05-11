"""Run a single KXBTC15M Markov-chain pipeline cycle and print the result.

Usage:
    python3 scan_btc15m_once.py

This is the easiest way to verify the bot is wired up correctly without
spinning up the full FastAPI server. It runs one cycle of the same job
the scheduler uses, posts to Discord (if webhook is configured), and
records the signal to Postgres.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

# Make sure we can import `backend` when run from project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def preflight() -> bool:
    """Print env-var status. Return True if we have enough to run."""
    from backend.config import settings

    print("\n── Preflight ──")
    checks = [
        ("KXBTC15M_ENABLED",         settings.KXBTC15M_ENABLED, False),
        ("KALSHI_API_KEY_ID",        bool(settings.KALSHI_API_KEY_ID), True),
        ("KALSHI_PRIVATE_KEY_PATH",  bool(settings.KALSHI_PRIVATE_KEY_PATH), True),
        ("ANTHROPIC_API_KEY",        bool(settings.ANTHROPIC_API_KEY), False),
        ("DISCORD_BTC_WEBHOOK_URL",  bool(settings.DISCORD_BTC_WEBHOOK_URL), False),
    ]
    blocking = False
    for name, ok, required in checks:
        mark = "✓" if ok else ("✗" if required else "·")
        label = "REQUIRED" if required else "optional"
        print(f"  {mark} {name:<28} {label:<8} {'set' if ok else 'NOT set'}")
        if required and not ok:
            blocking = True
    if not settings.ANTHROPIC_API_KEY:
        print("  → ANTHROPIC_API_KEY missing: pipeline will use deterministic GBM fallback")
    if not settings.DISCORD_BTC_WEBHOOK_URL:
        print("  → DISCORD_BTC_WEBHOOK_URL missing: signal will be printed but not posted")
    print()
    return not blocking


async def main():
    pem_contents = os.getenv("KALSHI_PRIVATE_KEY_CONTENTS")
    pem_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "/app/kalshi-prod.pem")
    if pem_contents and not os.path.exists(pem_path):
        os.makedirs(os.path.dirname(pem_path), exist_ok=True)
        with open(pem_path, "w") as f:
            f.write(pem_contents)

    if not preflight():
        print("Missing required Kalshi credentials. Aborting.")
        sys.exit(2)

    # Ensure the btc15m_signals table exists.
    from backend.models.database import init_db
    init_db()

    from backend.btc15m.pipeline import run_pipeline
    from backend.btc15m.discord_alerter import Btc15mSignal, send_btc15m_alert
    from backend.models.database import SessionLocal, BotState, Btc15mSignal as Btc15mSignalRow
    from backend.config import settings

    db = SessionLocal()
    try:
        state = db.query(BotState).first()
        bankroll = state.bankroll if state else settings.INITIAL_BANKROLL
    finally:
        db.close()

    print(f"── Running Markov pipeline (bankroll=${bankroll:.2f}) ──\n")
    result = await run_pipeline(bankroll=bankroll)

    print("\n── Result ──")

    # Fields to display in order; skip gracefully if absent.
    _FIELDS = [
        ("recommendation",   "Recommendation"),
        ("reason",           "Reason"),
        ("ticker",           "Ticker"),
        ("strike",           "Strike"),
        ("spot",             "Spot"),
        ("time_left_seconds","Time left (s)"),
        ("yes_ask",          "YES ask"),
        ("no_ask",           "NO ask"),
        ("p_yes",            "Model P(YES)"),
        ("gap",              "Gap"),
        ("persistence",      "Persistence"),
        ("kelly_contracts",  "Kelly contracts"),
        ("hurst",            "Hurst exponent"),
        ("gk_vol",           "GK volatility"),
        ("risk_reasons",     "Risk reasons"),
    ]

    # Flatten market fields into the display dict.
    display: dict = dict(result)
    market = result.get("market")
    if market is not None:
        display.setdefault("ticker",           market.market_ticker)
        display.setdefault("strike",           market.floor_strike)
        display.setdefault("yes_ask",          market.yes_ask)
        display.setdefault("no_ask",           market.no_ask)
        display.setdefault("time_left_seconds", market.seconds_to_expiry)
    display.setdefault("kelly_contracts", display.get("contracts"))
    display.setdefault("gap", (
        abs(result["p_yes"] - 0.50) if "p_yes" in result else None
    ))

    for key, label in _FIELDS:
        val = display.get(key)
        if val is None:
            continue
        if key == "strike":
            print(f"  {label:<20} ${val:,.2f}")
        elif key == "spot":
            print(f"  {label:<20} ${val:,.2f}")
        elif key in ("p_yes", "gap", "persistence", "hurst"):
            print(f"  {label:<20} {val:.3f}")
        elif key == "gk_vol":
            print(f"  {label:<20} {val:.4f}")
        elif key in ("yes_ask", "no_ask"):
            print(f"  {label:<20} {val:.3f}")
        elif key == "time_left_seconds":
            print(f"  {label:<20} {val:.0f}s ({val/60:.1f} min)")
        elif key == "risk_reasons" and isinstance(val, list):
            print(f"  {label:<20} {', '.join(val) if val else '—'}")
        else:
            print(f"  {label:<20} {val}")

    rec = result.get("recommendation", "PASS")
    if rec == "PASS":
        print("\n  No actionable signal this cycle.")
        return

    # Persist to DB.
    if market is not None:
        db = SessionLocal()
        try:
            row = Btc15mSignalRow(
                market_ticker   = market.market_ticker,
                event_ticker    = market.event_ticker,
                floor_strike    = market.floor_strike,
                close_time      = market.close_time,
                spot            = result.get("spot", 0.0),
                yes_ask         = market.yes_ask,
                yes_bid         = market.yes_bid,
                no_ask          = market.no_ask,
                no_bid          = market.no_bid,
                volume          = market.volume,
                market_implied_p= result.get("market_implied_p", market.market_implied_p),
                p_yes           = result.get("p_yes", 0.0),
                edge            = result.get("edge", 0.0),
                recommendation  = rec,
                confidence      = result.get("confidence", 0.0),
                contracts       = result.get("contracts", 0),
                notional_usd    = result.get("notional_usd", 0.0),
                kelly_fraction  = result.get("kelly_fraction", 0.0),
                thesis          = result.get("thesis"),
                risk_reasons    = result.get("risk_reasons", []),
                auto_executed   = False,
            )
            db.add(row)
            db.commit()
            print(f"\n  Saved signal #{row.id} to btc15m_signals.")
        finally:
            db.close()

    # Discord alert.
    signal = Btc15mSignal(
        recommendation   = rec,
        market           = market,
        spot             = result.get("spot", 0.0),
        p_yes            = result.get("p_yes", 0.0),
        edge             = result.get("edge", 0.0),
        market_implied_p = result.get("market_implied_p", 0.0),
        contracts        = result.get("contracts", 0),
        notional_usd     = result.get("notional_usd", 0.0),
        kelly_fraction   = result.get("kelly_fraction", 0.0),
        confidence       = result.get("confidence", 0.0),
        thesis           = result.get("thesis"),
        risk_reasons     = result.get("risk_reasons", []),
    )
    sent = await send_btc15m_alert(signal)
    print(f"  Discord alert: {'sent' if sent else 'skipped'}")


if __name__ == "__main__":
    asyncio.run(main())
