"""Signal generator for weather temperature markets using ensemble forecasts."""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import List, Optional

from backend.config import settings

from backend.notifications.discord import send_discord_signal
from backend.data.weather import fetch_ensemble_forecast
from backend.data.weather_markets import WeatherMarket
from backend.models.database import SessionLocal, Signal, LiveSignal

logger = logging.getLogger("trading_bot")

# Historical win rates keyed by (signal_type, no_price_cents_bucket)
HISTORICAL_WIN_RATES = {
    "T-above": {
        (85, 99): 0.994,
        (70, 84): 0.899,
        (55, 69): 0.688,
    },
    "B-above": {
        (85, 99): 0.990,
        (70, 84): 0.916,
        (55, 69): 0.850,
    },
    "B-below": {
        (85, 99): 0.990,
        (70, 84): 0.918,
        (55, 69): 0.897,
    },
}


def get_historical_win_rate(signal_type: str, no_price: float) -> float:
    no_price_cents = round(no_price * 100)
    buckets = HISTORICAL_WIN_RATES.get(signal_type, {})
    for (low, high), win_rate in buckets.items():
        if low <= no_price_cents <= high:
            return win_rate
    return 0.0


def calculate_position_size(
    historical_win_rate: float,
    expected_value: float,
    gfs_distance: float,
    yes_price: float,
    bankroll: float = 80.0,
) -> float:
    win_rate_score = historical_win_rate
    ev_score = min(expected_value / 0.30, 1.0)
    distance_score = min(gfs_distance / 10.0, 1.0)
    price_score = max(0.0, 1.0 - (yes_price / 0.30))

    confidence = (
        0.40 * win_rate_score +
        0.35 * ev_score +
        0.15 * distance_score +
        0.10 * price_score
    )

    max_position = min(bankroll * 0.225, 50.0)
    min_position = 10.0

    position_size = min_position + (confidence * (max_position - min_position))
    return round(position_size)


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

    # Expected value as fraction (e.g. 0.15 = 15% EV)
    expected_value: float = 0.0
    hist_win_rate: float = 0.0
    yes_price_cents: int = 0
    no_price_cents: int = 0
    signal_type: str = ""
    models_used: List[str] = field(default_factory=list)
    current_metar_high: Optional[float] = None

    @property
    def passes_threshold(self) -> bool:
        return self.edge >= 3.0 and self.expected_value > 0


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
    if not forecast or not forecast.member_highs:
        logger.debug(f"No forecast available for {market.market_id} ({market.city_key} on {market.target_date}) — skipping")
        return None

    # Ensemble stats
    mean_val = forecast.mean_high if market.metric == "high" else forecast.mean_low
    std_val  = forecast.std_high  if market.metric == "high" else forecast.std_low

    # Determine signal type and temperature-based edge
    if market.direction == "below":
        signal_type = "T-above"
        edge_f = mean_val - market.threshold_f
        if mean_val <= market.threshold_f + 3.0:
            logger.info(f"SKIP {market.market_id}: T-above MT {mean_val:.1f}F within 3.0F of threshold {market.threshold_f:.0f}F")
            return None
    elif market.direction == "above":
        if mean_val > market.threshold_f + 0.5:
            signal_type = "B-above"
            edge_f = mean_val - (market.threshold_f + 0.5)
            if edge_f < 1.0:
                logger.info(f"SKIP {market.market_id}: B-above edge {edge_f:.1f}F below 1.0F minimum")
                return None
        elif mean_val < market.threshold_f - 0.5:
            signal_type = "B-below"
            bottom_of_range = market.threshold_f - 0.5
            edge_f = bottom_of_range - mean_val
            if mean_val >= bottom_of_range - 3.0:
                logger.info(f"SKIP {market.market_id}: B-below MT {mean_val:.1f}F within 3.0F of bracket bottom {bottom_of_range:.1f}F")
                return None
        else:
            logger.info(f"SKIP {market.market_id}: MT {mean_val:.1f}F within bracket range [{market.threshold_f - 0.5:.1f}–{market.threshold_f + 0.5:.1f}F]")
            return None
    else:
        logger.info(f"SKIP {market.market_id}: unknown direction '{market.direction}'")
        return None

    # Always trade NO
    direction = "no"
    no_price = market.no_price
    yes_price_cents = round(market.yes_price * 100)
    no_price_cents = round(market.no_price * 100)

    # Entry price filters
    if market.yes_price > 0.30:
        logger.info(f"SKIP {market.market_id}: YES price {market.yes_price:.0%} > 30% (NO too cheap for edge)")
        return None
    if no_price > settings.WEATHER_MAX_ENTRY_PRICE:
        logger.info(f"SKIP {market.market_id}: NO price {no_price:.0%} above max entry {settings.WEATHER_MAX_ENTRY_PRICE:.0%}")
        return None

    # Historical win rate and EV
    hist_win_rate  = get_historical_win_rate(signal_type, no_price)
    expected_value = hist_win_rate - no_price  # positive if hist WR beats NO cost

    # Ensemble probability (used for confidence and sizing)
    members = forecast.member_highs if market.metric == "high" else forecast.member_lows
    if not members:
        logger.info(f"SKIP {market.market_id}: no ensemble members in forecast")
        return None
    if market.direction == "above":
        low = market.threshold_f - 0.5
        high = market.threshold_f + 0.5
        model_yes_prob = len([m for m in members if low <= m <= high]) / len(members)
    else:
        model_yes_prob = len([m for m in members if m < market.threshold_f]) / len(members) if members else 0.5
    model_yes_prob = max(0.05, min(0.95, model_yes_prob))
    market_yes_prob = market.yes_price

    # Confidence = ensemble agreement
    above_count   = sum(1 for m in members if m > market.threshold_f)
    agreement_frac = max(above_count, len(members) - above_count) / len(members)
    confidence    = min(0.9, agreement_frac)

    bankroll = settings.INITIAL_BANKROLL
    suggested_size = calculate_position_size(
        historical_win_rate=hist_win_rate,
        expected_value=expected_value,
        gfs_distance=abs(edge_f),
        yes_price=market.yes_price,
        bankroll=bankroll,
    )

    signal = WeatherTradingSignal(
        market=market,
        model_probability=model_yes_prob,
        market_probability=market_yes_prob,
        edge=edge_f,
        direction=direction,
        confidence=confidence,
        kelly_fraction=suggested_size / bankroll if bankroll > 0 else 0,
        suggested_size=suggested_size,
        sources=[f"minutetemp_{forecast.num_members}m"],
        ensemble_mean=mean_val,
        ensemble_std=std_val,
        ensemble_members=forecast.num_members,
        expected_value=expected_value,
        hist_win_rate=hist_win_rate,
        yes_price_cents=yes_price_cents,
        no_price_cents=no_price_cents,
        signal_type=signal_type,
    )

    signal.models_used = getattr(forecast, "models_used", [])
    signal.current_metar_high = getattr(forecast, "current_metar_high", None)

    if not signal.passes_threshold:
        logger.info(f"SKIP {market.market_id}: passes_threshold=False (edge={edge_f:.1f}F EV={expected_value*100:.1f}%)")

    filter_status = "ACTIONABLE" if signal.passes_threshold else "FILTERED"
    signal.reasoning = (
        f"[{filter_status}] [{signal_type}] "
        f"{market.city_name} {market.metric} {market.direction} {market.threshold_f:.0f}F on {market.target_date} | "
        f"Ensemble: {mean_val:.1f}F +/- {std_val:.1f}F ({forecast.num_members} members) | "
        f"Edge: {edge_f:.1f}°F | EV: {expected_value*100:.1f}% | "
        f"NO cost: {no_price*100:.0f}¢ | Hist WR: {hist_win_rate*100:.1f}%"
    )

    return signal


def _persist_live_signal(signal: "WeatherTradingSignal") -> None:
    db = SessionLocal()
    try:
        today = date.today()
        existing = db.query(LiveSignal).filter(
            LiveSignal.ticker == signal.market.market_id,
            LiveSignal.signal_fired_at >= datetime(today.year, today.month, today.day,
                                                   tzinfo=timezone.utc),
        ).first()
        if existing:
            logger.debug(f"Duplicate signal skipped for {signal.market.market_id} — already logged today")
            return

        record = LiveSignal(
            ticker=signal.market.market_id,
            signal_type=signal.signal_type,
            city=signal.market.city_key,
            target_date=str(signal.market.target_date),
            threshold_f=signal.market.threshold_f,
            gfs_mean=signal.ensemble_mean,
            gfs_std=signal.ensemble_std,
            gfs_distance=abs(signal.edge),
            yes_price_cents=round(signal.market.yes_price * 100),
            no_price_cents=round(signal.market.no_price * 100),
            historical_win_rate=signal.hist_win_rate,
            expected_value=signal.expected_value,
            confidence_score=signal.confidence,
            suggested_size=signal.suggested_size,
            reasoning=signal.reasoning,
        )
        db.add(record)
        db.commit()
        logger.info(f"Persisted live signal for {signal.market.market_id}")
    except Exception as e:
        logger.warning(f"Failed to persist live signal: {e}")
        db.rollback()
    finally:
        db.close()


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
                logger.info(f"Cached forecast for {city_key} on {target_date}: {forecast.mean_high:.1f}F")
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
                     f"{signal.market.threshold_f:.0f}F | Edge: {signal.edge:.1f}°F | EV: {signal.expected_value*100:.1f}%")

    if not settings.TRADING_ENABLED:
        logger.info("TRADING DISABLED — signal only mode")

    # Persist actionable signals to live_signals table
    for signal in actionable:
        _persist_live_signal(signal)

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
                "confidence": signal.confidence,
                "market": {"threshold_f": signal.market.threshold_f, "direction": signal.market.direction},
                "hist_win_rate": signal.hist_win_rate,
                "yes_price_cents": signal.yes_price_cents,
                "no_price_cents": signal.no_price_cents,
                "models_used": signal.models_used,
                "current_metar_high": signal.current_metar_high,
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
