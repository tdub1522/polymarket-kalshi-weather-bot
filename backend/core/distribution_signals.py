from datetime import date, datetime, timezone
from typing import List, Optional, Tuple
import logging
import statistics
from scipy.stats import norm

from backend.config import settings
from backend.data.weather import fetch_ensemble_forecast
from backend.data.kalshi_markets import fetch_kalshi_weather_markets
from backend.notifications.discord import send_distribution_signal

logger = logging.getLogger("trading_bot")

MAX_STD = 1.5           # Maximum std allowed for signal generation
MAX_COMBINED_YES = 0.90 # Maximum combined YES price for inside brackets
MAX_OUTSIDE_YES = 0.15  # Maximum YES price for outside contracts (NO >= 85c)
SIGMA_MULTIPLIER = 2.0  # Number of standard deviations for range


def get_brackets_in_range(
    markets: list,
    lower_bound: float,
    upper_bound: float,
    target_date: date,
    city_key: str,
) -> list:
    """
    Find all bracket (B-type) markets for a city/date
    whose full range falls inside [lower_bound, upper_bound].
    """
    inside = []
    for m in markets:
        if m.city_key != city_key:
            continue
        if m.target_date != target_date:
            continue
        if m.direction != "above":
            continue
        bracket_low = m.threshold_f - 0.5
        bracket_high = m.threshold_f + 0.5
        if bracket_low >= lower_bound and bracket_high <= upper_bound:
            inside.append(m)
    return inside


def get_outside_brackets(
    markets: list,
    lower_bound: float,
    upper_bound: float,
    target_date: date,
    city_key: str,
    mean: float,
) -> Tuple[Optional[object], Optional[object]]:
    """
    Find the bracket markets immediately outside the 2sigma range.
    Returns (lower_outside, upper_outside) — either can be None.
    Only returns brackets with YES <= MAX_OUTSIDE_YES.
    """
    lower_outside = None
    upper_outside = None

    for m in markets:
        if m.city_key != city_key:
            continue
        if m.target_date != target_date:
            continue
        if m.direction != "above":
            continue
        if m.yes_price > MAX_OUTSIDE_YES:
            continue

        bracket_high = m.threshold_f + 0.5
        bracket_low = m.threshold_f - 0.5

        # Just below lower bound
        if bracket_high <= lower_bound:
            if lower_outside is None or bracket_high > (lower_outside.threshold_f + 0.5):
                lower_outside = m

        # Just above upper bound
        if bracket_low >= upper_bound:
            if upper_outside is None or bracket_low < (upper_outside.threshold_f - 0.5):
                upper_outside = m

    return lower_outside, upper_outside


def pick_best_outside(
    lower_outside: Optional[object],
    upper_outside: Optional[object],
    mean: float,
) -> List[object]:
    """
    Pick the outside contract farthest from the mean.
    If both qualify and equidistant, return both.
    """
    if not lower_outside and not upper_outside:
        return []
    if lower_outside and not upper_outside:
        return [lower_outside]
    if upper_outside and not lower_outside:
        return [upper_outside]

    lower_dist = abs(mean - (lower_outside.threshold_f + 0.5))
    upper_dist = abs(mean - (upper_outside.threshold_f - 0.5))

    if abs(lower_dist - upper_dist) < 0.5:
        return [lower_outside, upper_outside]
    elif lower_dist > upper_dist:
        return [lower_outside]
    else:
        return [upper_outside]


async def scan_distribution_signals() -> List[dict]:
    """
    Scan for normal distribution edge opportunities.
    Only looks at today's contracts.
    """
    signals = []
    today = date.today()

    city_keys = [c.strip() for c in settings.WEATHER_CITIES.split(",") if c.strip()]

    logger.info("=" * 50)
    logger.info("DISTRIBUTION SCAN: Fetching today's markets...")

    # Fetch all markets
    try:
        from backend.data.kalshi_client import kalshi_credentials_present
        if not kalshi_credentials_present():
            logger.warning("Kalshi credentials not present")
            return []
        markets = await fetch_kalshi_weather_markets(city_keys)
    except Exception as e:
        logger.error(f"Failed to fetch markets for distribution scan: {e}")
        return []

    # Filter to today only
    today_markets = [m for m in markets if m.target_date == today]
    logger.info(f"Distribution scan: {len(today_markets)} markets for today ({today})")

    if not today_markets:
        logger.info("No today markets found for distribution scan")
        return []

    # Get unique city keys for today
    city_keys_today = list(set(m.city_key for m in today_markets))

    for city_key in city_keys_today:
        try:
            # Fetch forecast
            forecast = await fetch_ensemble_forecast(city_key, today)
            if not forecast:
                logger.debug(f"No forecast for {city_key}")
                continue

            mean = forecast.mean_high
            std = forecast.std_high
            num_members = forecast.num_members

            # Std filter
            if std >= MAX_STD:
                logger.info(f"DIST SKIP {city_key}: std {std:.2f}F >= {MAX_STD}F threshold")
                continue

            if std == 0:
                logger.info(f"DIST SKIP {city_key}: std is 0")
                continue

            # Calculate 2-sigma range
            lower_bound = mean - (SIGMA_MULTIPLIER * std)
            upper_bound = mean + (SIGMA_MULTIPLIER * std)
            theoretical_prob = norm.cdf(upper_bound, mean, std) - norm.cdf(lower_bound, mean, std)

            logger.info(
                f"DIST {city_key}: mean={mean:.1f}F std={std:.2f}F "
                f"2σ=[{lower_bound:.1f}, {upper_bound:.1f}] "
                f"P(inside)={theoretical_prob*100:.1f}%"
            )

            # Find inside brackets
            inside_brackets = get_brackets_in_range(
                today_markets, lower_bound, upper_bound, today, city_key
            )

            if not inside_brackets:
                logger.info(f"DIST SKIP {city_key}: no brackets inside 2σ range")
                continue

            # Sum combined YES of inside brackets
            combined_yes = sum(m.yes_price for m in inside_brackets)

            logger.info(
                f"DIST {city_key}: {len(inside_brackets)} inside brackets, "
                f"combined YES={combined_yes*100:.0f}¢"
            )

            if combined_yes > MAX_COMBINED_YES:
                logger.info(
                    f"DIST SKIP {city_key}: combined YES {combined_yes*100:.0f}¢ "
                    f"> {MAX_COMBINED_YES*100:.0f}¢ threshold"
                )
                continue

            # Edge = theoretical probability - combined YES price
            edge = theoretical_prob - combined_yes
            logger.info(
                f"DIST {city_key}: edge={edge*100:.1f}¢ "
                f"(theory={theoretical_prob*100:.1f}% vs market={combined_yes*100:.0f}¢)"
            )

            # Find outside brackets
            lower_outside, upper_outside = get_outside_brackets(
                today_markets, lower_bound, upper_bound, today, city_key, mean
            )

            targets = pick_best_outside(lower_outside, upper_outside, mean)

            if not targets:
                logger.info(f"DIST SKIP {city_key}: no qualifying outside contracts (YES <= {MAX_OUTSIDE_YES*100:.0f}¢)")
                continue

            for target in targets:
                signal = {
                    "city_key": city_key,
                    "city_name": target.city_name,
                    "target_date": str(today),
                    "mean": mean,
                    "std": std,
                    "num_members": num_members,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "theoretical_prob": theoretical_prob,
                    "inside_brackets": [m.market_id for m in inside_brackets],
                    "combined_yes": combined_yes,
                    "edge": edge,
                    "target_ticker": target.market_id,
                    "target_yes_price": target.yes_price,
                    "target_no_price": target.no_price,
                    "target_threshold": target.threshold_f,
                    "side": "no",
                    "suggested_size": 15.0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                signals.append(signal)
                logger.info(
                    f"DIST SIGNAL {city_key}: BUY NO on {target.market_id} "
                    f"YES={target.yes_price*100:.0f}¢ NO={target.no_price*100:.0f}¢"
                )

        except Exception as e:
            logger.warning(f"Distribution scan failed for {city_key}: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            continue

    logger.info(f"DISTRIBUTION SCAN COMPLETE: {len(signals)} signals")

    # Send to Discord
    if settings.DISCORD_ENABLED and settings.DISCORD_DISTRIBUTION_WEBHOOK_URL:
        for signal in signals:
            await send_distribution_signal(signal)

    return signals
