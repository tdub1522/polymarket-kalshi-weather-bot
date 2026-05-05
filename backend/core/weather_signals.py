"""Signal generator for weather temperature markets using ensemble forecasts."""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import List, Optional

from backend.config import settings
from backend.core.signals import calculate_edge, calculate_kelly_size
from backend.notifications.discord import send_discord_signal
from backend.data.weather import fetch_ensemble_forecast
from backend.data.weather_markets import WeatherMarket
from backend.models.database import SessionLocal, Signal

logger = logging.getLogger("trading_bot")


@dataclass
class WeatherTradingSignal:
    """A trading signal for a weather temperature market."""
    market: WeatherMarket

    # Core signal data
    model_probability: float = 0.5   # Ensemble probability of YES outcome
    market_probability: float = 0.5  # Market's implied YES probability
    edge: float = 0.0
    direction: str = "yes"           # "yes" or "no"

    # Confidence and sizing
    confidence: float = 0.5
    kelly_fraction: float = 0.0
    suggested_size: float = 0.0

    # Metadata
    sources: List[str] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Forecast context
    ensemble_mean: float = 0.0
    ensemble_std: float = 0.0
    ensemble_members: int = 0

    # Expected value in percentage points (e.g. 15.0 = 15% EV)
    expected_value: float = 0.0

    @property
    def passes_threshold(self) -> bool:
        return (
            abs(self.edge) >= settings.WEATHER_MIN_EDGE_THRESHOLD
            and self.model_probability <= 0.10
            and self.expected_value > 0
        )


async def generate_weather_signal(
    market: WeatherMarket,
    forecast=None,
) -> Optional[WeatherTradingSignal]:
    """
    Generate a trading signal for a weather temperature market.

    Uses ensemble forecast to estimate probability:
    - Count fraction of ensemble members above/below the threshold
    - Compare to market price to find edge
    - Size using Kelly criterion
    """
    if forecast is None:
        forecast = await fetch_ensemble_forecast(market.city_key, market.target_date)
    if not forecast or not forecast.member_highs:
        return None

    # Get ensemble members for this market's metric (used for both probability and confidence)
    members = forecast.member_highs if market.metric == "high" else forecast.member_lows

    # Calculate model probability based on market type
    if market.direction == "above":
        # Bracket market (B-type ticker): probability within ±0.5°F of threshold
        low = market.threshold_f - 0.5
        high = market.threshold_f + 0.5
        model_yes_prob = len([m for m in members if low <= m <= high]) / len(members)
    else:
        # Non-bracket market (T-type): probability that temp stays below threshold
        if market.metric == "high":
            model_yes_prob = forecast.probability_high_below(market.threshold_f)
        else:
            model_yes_prob = forecast.probability_low_below(market.threshold_f)

    # Clip extreme probabilities (ensemble can be unanimous but don't bet 100%)
    model_yes_prob = max(0.05, min(0.95, model_yes_prob))

    market_yes_prob = market.yes_price

    # Use existing edge calculation (treats yes=up, no=down)
    edge, direction_raw = calculate_edge(model_yes_prob, market_yes_prob)
    direction = "yes" if direction_raw == "up" else "no"

    # Only trade NO signals — YES signals not validated by backtest
    if direction == "yes":
        logger.debug(f"Skipping {market.market_id} — YES signals not supported by backtest")
        return None

    # Entry price filter — check the price of the side we're trading
    entry_price = market.no_price if direction == "no" else market.yes_price
    if entry_price > settings.WEATHER_MAX_ENTRY_PRICE:
        edge = 0.0  # Zero out but still return for UI visibility

    # Expected value calculation
    model_no_prob = 1 - model_yes_prob
    if direction == "yes":
        ev = (model_yes_prob * (1 - market.yes_price)) - ((1 - model_yes_prob) * market.yes_price)
    else:
        no_price = 1 - market.yes_price
        ev = (model_no_prob * (1 - no_price)) - ((1 - model_no_prob) * no_price)
    expected_value = ev * 100

    # Confidence = ensemble agreement (how one-sided the members are)
    above_count = sum(1 for m in members if m > market.threshold_f)
    agreement_frac = max(above_count, len(members) - above_count) / len(members)
    confidence = min(0.9, agreement_frac)

    # Kelly sizing
    bankroll = settings.INITIAL_BANKROLL
    suggested_size = calculate_kelly_size(
        edge=abs(edge),
        probability=model_yes_prob,
        market_price=market_yes_prob,
        direction=direction_raw,  # calculate_kelly_size expects "up"/"down"
        bankroll=bankroll,
    )
    suggested_size = min(suggested_size, settings.WEATHER_MAX_TRADE_SIZE)

    # Ensemble stats for display
    mean_val = forecast.mean_high if market.metric == "high" else forecast.mean_low
    std_val = forecast.std_high if market.metric == "high" else forecast.std_low

    # GFS distance filter — only T-contracts where GFS is clearly above threshold
    if market.direction == "above":
        low = market.threshold_f - 0.5
        high = market.threshold_f + 0.5
        if (mean_val > low - 1.0) and (mean_val < high + 1.0):
            logger.debug(f"Skipping {market.market_id} — GFS mean {mean_val:.1f}F too close to range {low:.0f}-{high:.0f}F")
            return None
    else:
        # Require GFS mean to be >= 3.5F above threshold (validated sweet spot)
        # Also skip if GFS predicts YES (mean_val < threshold_f)
        if mean_val < market.threshold_f or (mean_val - market.threshold_f) < 3.5:
            logger.debug(f"Skipping {market.market_id} — GFS mean {mean_val:.1f}F not >= 3.5F above threshold {market.threshold_f:.0f}F")
            return None

    if std_val > 1.0:
        logger.debug(f"Skipping {market.market_id} — GFS std {std_val:.1f}F exceeds 1.0F threshold")
        return None

    # Build reasoning
    filter_status = "ACTIONABLE" if abs(edge) >= settings.WEATHER_MIN_EDGE_THRESHOLD else "FILTERED"
    filter_notes = []
    if entry_price > settings.WEATHER_MAX_ENTRY_PRICE:
        filter_notes.append(f"entry {entry_price:.0%} > {settings.WEATHER_MAX_ENTRY_PRICE:.0%}")
    filter_note = f" [{', '.join(filter_notes)}]" if filter_notes else ""

    reasoning = (
        f"[{filter_status}]{filter_note} "
        f"{market.city_name} {market.metric} {market.direction} {market.threshold_f:.0f}F on {market.target_date} | "
        f"Ensemble: {mean_val:.1f}F +/- {std_val:.1f}F ({forecast.num_members} members) | "
        f"Model YES: {model_yes_prob:.0%} vs Market: {market_yes_prob:.0%} | "
        f"Edge: {edge:+.1%} | EV: {expected_value:+.1f}% -> {direction.upper()} @ {entry_price:.0%} | "
        f"Agreement: {agreement_frac:.0%}"
    )

    return WeatherTradingSignal(
        market=market,
        model_probability=model_yes_prob,
        market_probability=market_yes_prob,
        edge=edge,
        direction=direction,
        confidence=confidence,
        kelly_fraction=suggested_size / bankroll if bankroll > 0 else 0,
        suggested_size=suggested_size,
        sources=[f"open_meteo_ensemble_{forecast.num_members}m"],
        reasoning=reasoning,
        ensemble_mean=mean_val,
        ensemble_std=std_val,
        ensemble_members=forecast.num_members,
        expected_value=expected_value,
    )


async def scan_for_weather_signals() -> List[WeatherTradingSignal]:
    """
    Scan weather markets and generate ensemble-based signals.
    """
    signals = []

    city_keys = [c.strip() for c in settings.WEATHER_CITIES.split(",") if c.strip()]

    logger.info("=" * 50)
    logger.info("WEATHER SCAN: Fetching temperature markets...")

    markets = []

    # Kalshi
    if settings.KALSHI_ENABLED:
        try:
            from backend.data.kalshi_client import kalshi_credentials_present
            from backend.data.kalshi_markets import fetch_kalshi_weather_markets
            if kalshi_credentials_present():
                kalshi_markets = await fetch_kalshi_weather_markets(city_keys)
                markets.extend(kalshi_markets)
                logger.info(f"Kalshi: {len(kalshi_markets)} weather markets")
        except Exception as e:
            logger.error(f"Failed to fetch Kalshi weather markets: {e}")

    logger.info(f"Found {len(markets)} total weather temperature markets")

    # Pre-fetch ensemble forecasts keyed by (city_key, target_date)
    # so tomorrow's contracts use tomorrow's GFS forecast, not today's
    forecast_cache = {}

    city_date_pairs = set()
    for market in markets:
        city_date_pairs.add((market.city_key, market.target_date))

    for city_key, target_date in city_date_pairs:
        cache_key = (city_key, target_date)
        try:
            forecast = await fetch_ensemble_forecast(city_key, target_date)
            if forecast:
                forecast_cache[cache_key] = forecast
        except Exception as e:
            logger.warning(f"Failed to pre-fetch ensemble for {city_key} {target_date}: {e}")

    logger.info(f"Ensemble cache populated for {len(forecast_cache)}/{len(city_date_pairs)} city+date pairs")

    for market in markets:
        try:
            cache_key = (market.city_key, market.target_date)
            signal = await generate_weather_signal(market, forecast=forecast_cache.get(cache_key))
            if signal:
                signals.append(signal)
        except Exception as e:
            logger.debug(f"Weather signal generation failed for {market.title}: {e}")

    # Sort by absolute edge
    signals.sort(key=lambda s: abs(s.edge), reverse=True)

    actionable = [s for s in signals if s.passes_threshold]
    logger.info(f"WEATHER SCAN COMPLETE: {len(signals)} signals, {len(actionable)} actionable")

    for signal in actionable[:5]:
        logger.info(f"  {signal.market.city_name}: {signal.market.metric} {signal.market.direction} "
                     f"{signal.market.threshold_f:.0f}F | Edge: {signal.edge:+.1%}")

    if not settings.TRADING_ENABLED:
        logger.info("TRADING DISABLED — signal only mode")

    # Notify Discord — only when alerts are enabled and auto-trading is explicitly off
    if settings.DISCORD_ENABLED and not settings.TRADING_ENABLED:
        for signal in actionable:
            await send_discord_signal({
                "market_title": signal.market.title,
                "ticker": signal.market.market_id,
                "side": signal.direction,
                "edge": signal.edge,
                "expected_value": signal.expected_value,
                "yes_price": signal.market.yes_price,
                "no_price": signal.market.no_price,
                "model_probability": signal.model_probability,
                "suggested_size": signal.suggested_size,
                "ensemble_members": signal.ensemble_members,
                "ensemble_mean": signal.ensemble_mean,
                "ensemble_std": signal.ensemble_std,
                "market": {"threshold_f": signal.market.threshold_f, "direction": signal.market.direction},
            })

    # Persist signals to DB
    _persist_weather_signals(signals)

    return signals


def _persist_weather_signals(signals: list):
    """Save weather signals to DB for calibration tracking."""
    to_save = [s for s in signals if abs(s.edge) > 0]
    if not to_save:
        return

    db = SessionLocal()
    try:
        for signal in to_save:
            # Dedup: skip if already logged for this market
            existing = db.query(Signal).filter(
                Signal.market_ticker == signal.market.market_id,
                Signal.timestamp >= signal.timestamp.replace(second=0, microsecond=0),
            ).first()
            if existing:
                continue

            db_signal = Signal(
                market_ticker=signal.market.market_id,
                platform=signal.market.platform,
                market_type="weather",
                timestamp=signal.timestamp,
                direction=signal.direction,
                model_probability=signal.model_probability,
                market_price=signal.market_probability,
                edge=signal.edge,
                confidence=signal.confidence,
                kelly_fraction=signal.kelly_fraction,
                suggested_size=signal.suggested_size,
                sources=signal.sources,
                reasoning=signal.reasoning,
                executed=False,
            )
            db.add(db_signal)

        db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist weather signals: {e}")
        db.rollback()
    finally:
        db.close()
