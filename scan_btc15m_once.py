"""Run a single KXBTC15M pipeline cycle and print the result.

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
    if not preflight():
        print("Missing required Kalshi credentials. Aborting.")
        sys.exit(2)

    # Ensure the btc15m_signals table exists.
    from backend.models.database import init_db
    init_db()

    from backend.btc15m.pipeline import run_btc15m_pipeline
    from backend.btc15m.discord_alerter import send_btc15m_alert
    from backend.models.database import SessionLocal, BotState, Btc15mSignal
    from backend.config import settings

    db = SessionLocal()
    try:
        state = db.query(BotState).first()
        bankroll = state.bankroll if state else settings.INITIAL_BANKROLL
    finally:
        db.close()

    print(f"── Running pipeline (bankroll=${bankroll:.2f}) ──\n")
    result = await run_btc15m_pipeline(
        bankroll=bankroll,
        daily_pnl=0.0,
        drawdown_pct=0.0,
        trades_today=0,
    )

    print(f"\n── Result ── ({result.elapsed_ms:.0f}ms, {result.markets_seen} markets seen)")
    if result.error:
        print(f"  ERROR: {result.error}")
        return
    if not result.signal:
        print("  No signal this cycle (no tradeable strike near spot, or markets closed).")
        return

    sig = result.signal
    m = sig.market
    print(f"  Ticker:       {m.market_ticker}")
    print(f"  Strike:       ${m.floor_strike:,.2f}")
    print(f"  Spot:         ${sig.spot:,.2f}")
    print(f"  Time left:    {m.seconds_to_expiry/60:.1f} min")
    print(f"  YES bid/ask:  {m.yes_bid:.3f} / {m.yes_ask:.3f}")
    print(f"  NO  bid/ask:  {m.no_bid:.3f} / {m.no_ask:.3f}")
    print()
    print(f"  Sentiment:        {sig.sentiment.label} ({sig.sentiment.score:+.2f}) "
          f"momentum={sig.sentiment.momentum}")
    print(f"  Model P(YES):     {sig.p_yes*100:.2f}%")
    print(f"  Market P(YES):    {sig.market_implied_p*100:.2f}%")
    print(f"  Edge:             {sig.edge*100:+.2f}%")
    print(f"  Confidence:       {sig.confidence*100:.0f}%")
    print(f"  Recommendation:   {sig.recommendation}")
    if sig.recommendation != "PASS":
        print(f"  Suggested size:   {sig.contracts} contracts (~${sig.notional_usd:.2f}, "
              f"Kelly={sig.kelly_fraction:.3f})")
    print(f"  Risk reasons:     {', '.join(sig.risk_reasons)}")
    if sig.thesis:
        print(f"\n  Thesis:\n    {sig.thesis[:600]}")

    # Persist to DB (matches the scheduler's path).
    db = SessionLocal()
    try:
        row = Btc15mSignal(
            market_ticker=m.market_ticker,
            event_ticker=m.event_ticker,
            floor_strike=m.floor_strike,
            close_time=m.close_time,
            spot=sig.spot,
            yes_ask=m.yes_ask,
            yes_bid=m.yes_bid,
            no_ask=m.no_ask,
            no_bid=m.no_bid,
            volume=m.volume,
            market_implied_p=sig.market_implied_p,
            p_yes=sig.p_yes,
            edge=sig.edge,
            recommendation=sig.recommendation,
            confidence=sig.confidence,
            contracts=sig.contracts,
            notional_usd=sig.notional_usd,
            kelly_fraction=sig.kelly_fraction,
            sentiment_label=sig.sentiment.label,
            sentiment_score=sig.sentiment.score,
            thesis=sig.thesis,
            key_drivers=sig.key_drivers,
            risk_reasons=sig.risk_reasons,
            auto_executed=False,
        )
        db.add(row)
        db.commit()
        print(f"\n  Saved signal #{row.id} to btc15m_signals.")
    finally:
        db.close()

    sent = await send_btc15m_alert(sig)
    print(f"  Discord alert: {'sent' if sent else 'skipped'}")


if __name__ == "__main__":
    asyncio.run(main())
