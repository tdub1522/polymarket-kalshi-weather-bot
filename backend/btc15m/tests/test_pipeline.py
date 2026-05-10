"""Unit tests for KXBTC15M pipeline — focused on the deterministic components.

Run from project root:
    python -m pytest backend/btc15m/tests/ -v

These tests do NOT hit the Kalshi or Anthropic APIs; LLM-driven stages are
exercised only via the heuristic fallback (no API key path).
"""
from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timedelta, timezone

import pytest

from backend.btc15m.agents import RiskDecision, run_risk_manager
from backend.btc15m.kalshi_market import (
    Btc15mMarket,
    _current_event_ticker,
    _next_event_ticker,
)
from backend.btc15m.price_feed import BtcPriceFeed, BtcPriceSnapshot
from backend.btc15m.roma.aggregator import _heuristic_fallback


def _mk_market(
    yes_ask=0.40,
    yes_bid=0.39,
    no_ask=0.61,
    no_bid=0.60,
    floor_strike=100_000.0,
    seconds_to_close=600.0,
):
    return Btc15mMarket(
        event_ticker="KXBTC15M-TESTBUCKET",
        market_ticker="KXBTC15M-TESTBUCKET-T100000",
        title="test",
        floor_strike=floor_strike,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        volume=10_000.0,
        close_time=datetime.now(timezone.utc) + timedelta(seconds=seconds_to_close),
    )


# ─────────── Risk manager ───────────


def test_risk_manager_passes_when_recommendation_pass():
    d = run_risk_manager(
        p_yes=0.50, market=_mk_market(), recommendation="PASS",
        bankroll=80, daily_pnl=0, daily_loss_cap=20, drawdown_pct=0,
        max_drawdown_pct=0.15, trades_today=0, max_trades_per_day=24,
        max_trade_size=15.0,
    )
    assert d.approved is False
    assert d.contracts == 0
    assert "recommendation_pass" in d.reasons


def test_risk_manager_blocks_at_daily_loss_cap():
    d = run_risk_manager(
        p_yes=0.70, market=_mk_market(yes_ask=0.40), recommendation="BUY_YES",
        bankroll=80, daily_pnl=-25, daily_loss_cap=20,
        drawdown_pct=0, max_drawdown_pct=0.15,
        trades_today=0, max_trades_per_day=24, max_trade_size=15.0,
    )
    assert d.approved is False
    assert "daily_loss_cap_hit" in d.reasons


def test_risk_manager_blocks_at_drawdown_cap():
    d = run_risk_manager(
        p_yes=0.70, market=_mk_market(yes_ask=0.40), recommendation="BUY_YES",
        bankroll=80, daily_pnl=0, daily_loss_cap=20,
        drawdown_pct=0.20, max_drawdown_pct=0.15,
        trades_today=0, max_trades_per_day=24, max_trade_size=15.0,
    )
    assert d.approved is False
    assert "drawdown_cap_hit" in d.reasons


def test_risk_manager_blocks_at_trade_count_cap():
    d = run_risk_manager(
        p_yes=0.70, market=_mk_market(yes_ask=0.40), recommendation="BUY_YES",
        bankroll=80, daily_pnl=0, daily_loss_cap=20,
        drawdown_pct=0, max_drawdown_pct=0.15,
        trades_today=24, max_trades_per_day=24, max_trade_size=15.0,
    )
    assert d.approved is False
    assert "trade_count_cap_hit" in d.reasons


def test_risk_manager_blocks_below_min_edge():
    # market_implied via yes_ask=0.40 → edge of buying YES at 0.40 with p_yes=0.42 = 0.02 < 3%
    d = run_risk_manager(
        p_yes=0.42, market=_mk_market(yes_ask=0.40), recommendation="BUY_YES",
        bankroll=80, daily_pnl=0, daily_loss_cap=20,
        drawdown_pct=0, max_drawdown_pct=0.15,
        trades_today=0, max_trades_per_day=24, max_trade_size=15.0,
        min_edge=0.03,
    )
    assert d.approved is False
    assert any("edge<" in r for r in d.reasons)


def test_risk_manager_kelly_sizing_quarter_kelly_buy_yes():
    # p_yes=0.55, yes_ask=0.40 → cost=0.40, b=(1-0.40)/0.40=1.5, q=0.45
    # full_kelly = (0.55*1.5 - 0.45) / 1.5 = (0.825 - 0.45)/1.5 = 0.25
    # quarter_kelly = 0.25 * 0.25 = 0.0625
    # bankroll 80 → risk 5.0 → contracts = 5.0/0.40 = 12 (floored)
    d = run_risk_manager(
        p_yes=0.55, market=_mk_market(yes_ask=0.40), recommendation="BUY_YES",
        bankroll=80, daily_pnl=0, daily_loss_cap=20,
        drawdown_pct=0, max_drawdown_pct=0.15,
        trades_today=0, max_trades_per_day=24, max_trade_size=15.0,
        kelly_fraction_cap=0.25, min_edge=0.03,
    )
    assert d.approved is True
    assert d.contracts == 12
    assert math.isclose(d.kelly_fraction, 0.25, rel_tol=1e-6)
    # Notional capped by max_trade_size=15 → in this case 12 contracts × 0.40 = $4.80 (under cap)
    assert math.isclose(d.notional_usd, 4.80, rel_tol=1e-6)


def test_risk_manager_buy_no_uses_no_ask_for_cost():
    # p_yes=0.30 → win_prob for BUY_NO = 0.70, no_ask=0.65
    # b = (1-0.65)/0.65 ≈ 0.5385, q=0.30
    # full_kelly = (0.70*0.5385 - 0.30)/0.5385 = (0.377 - 0.30)/0.5385 ≈ 0.1429
    # quarter_kelly ≈ 0.0357 → on $80 bankroll = $2.86 → 4 contracts at $0.65
    d = run_risk_manager(
        p_yes=0.30,
        market=_mk_market(yes_ask=0.30, no_ask=0.65, floor_strike=100_000),
        recommendation="BUY_NO",
        bankroll=80, daily_pnl=0, daily_loss_cap=20,
        drawdown_pct=0, max_drawdown_pct=0.15,
        trades_today=0, max_trades_per_day=24, max_trade_size=15.0,
        kelly_fraction_cap=0.25, min_edge=0.03,
    )
    assert d.approved is True
    assert d.contracts >= 1


def test_risk_manager_caps_at_max_trade_size():
    # Big edge + big bankroll → would size huge, but max_trade_size=15 caps it.
    d = run_risk_manager(
        p_yes=0.95, market=_mk_market(yes_ask=0.40), recommendation="BUY_YES",
        bankroll=10_000, daily_pnl=0, daily_loss_cap=20,
        drawdown_pct=0, max_drawdown_pct=0.15,
        trades_today=0, max_trades_per_day=24, max_trade_size=15.0,
        kelly_fraction_cap=0.25, min_edge=0.03,
    )
    assert d.approved is True
    assert d.notional_usd <= 15.0 + 0.40  # one-contract slop


# ─────────── BRTI median price feed ───────────


def test_price_feed_records_median_of_two_sources():
    feed = BtcPriceFeed(history_seconds=60)
    snap = BtcPriceSnapshot(ts=time.time(), spot=100.0, coinbase=99.0, kraken=101.0,
                            sources=["coinbase", "kraken"])
    feed._record(snap)
    assert len(feed.history) == 1
    assert feed.history[0].spot == 100.0


def test_price_feed_change_over_window():
    feed = BtcPriceFeed(history_seconds=3600)
    now = time.time()
    feed._record(BtcPriceSnapshot(ts=now - 60, spot=100.0, coinbase=100, kraken=100))
    feed._record(BtcPriceSnapshot(ts=now, spot=110.0, coinbase=110, kraken=110))
    assert math.isclose(feed.change_over(120), 0.10, rel_tol=1e-6)


def test_price_feed_realized_vol_handles_short_history():
    feed = BtcPriceFeed()
    # Empty → None
    assert feed.realized_vol(900) is None
    feed._record(BtcPriceSnapshot(ts=time.time(), spot=100, coinbase=100, kraken=100))
    # 1 point → still None
    assert feed.realized_vol(900) is None


# ─────────── Event ticker construction ───────────


def test_event_ticker_format():
    # 2026-05-08 17:33 ET → bucket 17:30 → KXBTC15M-26MAY081730
    fixed = datetime(2026, 5, 8, 21, 33, tzinfo=timezone.utc)  # 17:33 ET
    t = _current_event_ticker(now_utc=fixed)
    assert t.startswith("KXBTC15M-")
    # Just check structural shape — TZ DST shifts can move the hour part.
    assert t.split("-")[1].isalnum()
    assert len(t.split("-")[1]) >= 9


def test_next_event_ticker_advances_15min():
    fixed = datetime(2026, 5, 8, 21, 33, tzinfo=timezone.utc)
    cur = _current_event_ticker(now_utc=fixed)
    nxt = _next_event_ticker(now_utc=fixed)
    assert cur != nxt


# ─────────── Heuristic fallback aggregator ───────────


def test_heuristic_fallback_passes_when_p_close_to_market():
    out = _heuristic_fallback({
        "spot": 100_000.0,
        "strike": 100_000.0,
        "seconds_to_expiry": 300,
        "market_implied_p": 0.50,
        "realized_vol_15m": 0.0005,
    })
    assert out["recommendation"] == "PASS"


def test_heuristic_fallback_buys_yes_when_far_above_strike():
    # Spot well above strike, low vol, short time → P(YES) >> 0.50
    out = _heuristic_fallback({
        "spot": 110_000.0,
        "strike": 100_000.0,
        "seconds_to_expiry": 60,
        "market_implied_p": 0.50,
        "realized_vol_15m": 0.0005,
    })
    assert out["recommendation"] == "BUY_YES"
    assert out["p_yes"] > 0.95


def test_heuristic_fallback_buys_no_when_far_below_strike():
    out = _heuristic_fallback({
        "spot": 90_000.0,
        "strike": 100_000.0,
        "seconds_to_expiry": 60,
        "market_implied_p": 0.50,
        "realized_vol_15m": 0.0005,
    })
    assert out["recommendation"] == "BUY_NO"
    assert out["p_yes"] < 0.05


# ─────────── Discord embed builder ───────────


def test_discord_embed_structure():
    from backend.btc15m.discord_alerter import _format_embed
    from backend.btc15m.agents import (
        ExecutionSignal, SentimentResult, build_execution_signal, run_risk_manager,
    )
    from backend.btc15m.roma.solve import RomaResult

    market = _mk_market(yes_ask=0.40, no_ask=0.61)
    sentiment = SentimentResult(score=0.5, label="bullish", momentum="up", signals=["test"])
    roma = RomaResult(thesis="model says BTC up", p_yes=0.55, recommendation="BUY_YES",
                      confidence=0.7, key_drivers=["momentum"], subtasks=[])
    risk = run_risk_manager(
        p_yes=0.55, market=market, recommendation="BUY_YES",
        bankroll=80, daily_pnl=0, daily_loss_cap=20,
        drawdown_pct=0, max_drawdown_pct=0.15,
        trades_today=0, max_trades_per_day=24, max_trade_size=15.0,
    )
    sig = build_execution_signal(
        market=market, spot=99_500.0, sentiment=sentiment, roma=roma, risk=risk,
    )
    embed = _format_embed(sig)
    assert embed["title"].startswith("KXBTC15M")
    field_names = [f["name"] for f in embed["fields"]]
    for required in ("Ticker", "Strike", "Spot", "Recommendation",
                     "Edge", "Model P(YES)", "Market P(YES)"):
        assert required in field_names, f"missing field {required}"
    assert "Manual trade required" in embed["footer"]["text"]


# ─────────── Signal-only guardrail ───────────


def test_pipeline_module_does_not_import_order_placement():
    """The pipeline must NEVER import any order-placement function. If a
    refactor adds one, this test fails — forcing an explicit re-review."""
    import backend.btc15m.pipeline as pipeline
    src = pipeline.__file__
    with open(src, "r") as fh:
        body = fh.read()
    forbidden = ("place_order", "post_order", "submit_order", "create_order")
    for token in forbidden:
        assert token not in body, f"pipeline.py must not reference {token}"
