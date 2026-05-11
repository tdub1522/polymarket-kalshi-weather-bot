"""MinuteTemp API client — oracle-ranked ASOS weather forecasts and observations.

The oracle-score endpoint ranks models by recent accuracy; only models with
abs(high_bias) < 1.0 F are used when building the ensemble average, so the
signal reflects the most accurate models available for each station.
"""
from __future__ import annotations

import logging
import statistics
from datetime import date
from typing import Dict, List, Optional

import httpx

from backend.config import settings

logger = logging.getLogger("trading_bot")

BASE_URL = "https://api.minutetemp.com/api/v1"

CITY_STATION_MAP: Dict[str, dict] = {
    "nyc":          {"station_id": "KNYC", "slug": "nyc"},
    "chicago":      {"station_id": "KMDW", "slug": "chi"},
    "miami":        {"station_id": "KMIA", "slug": "mia"},
    "los_angeles":  {"station_id": "KLAX", "slug": "lax"},
    "denver":       {"station_id": "KDEN", "slug": "den"},
    "philadelphia": {"station_id": "KPHL", "slug": "phl"},
}

_API_HEADERS = lambda: {"X-API-Key": settings.MINUTETEMP_API_KEY or ""}


async def fetch_oracle_scores(station_id: str, mode: str) -> List[dict]:
    """Fetch oracle model scores for a station.

    mode: "day_of" or "day_ahead"
    Returns models with abs(high_bias) < 1.0, sorted by high_mae ascending.
    """
    params = {
        "rank_by": "high_mae",
        "mode":    mode,
        "window":  "7d",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{BASE_URL}/stations/{station_id}/oracle-scores",
            headers=_API_HEADERS(),
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    scores = data.get("data", {}).get("scores", [])
    qualified = [s for s in scores if abs(s.get("high_bias", 999)) < 1.0]

    logger.info(
        f"Oracle scores {station_id} ({mode}): "
        f"{len(qualified)}/{len(scores)} models qualify (|high_bias| < 1.0): "
        f"{[s['model_name'] for s in qualified]}"
    )
    return qualified


async def fetch_station_forecast(
    station_id: str,
    target_date: Optional[date] = None,
) -> Dict[str, List[float]]:
    """Fetch all model forecasts for a station.

    Returns dict keyed by model_id -> list of hourly temps (F) for target_date.
    If target_date is None, all hourly temps are returned.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{BASE_URL}/stations/{station_id}/forecast",
            headers=_API_HEADERS(),
        )
        resp.raise_for_status()
        data = resp.json()

    model_forecasts: Dict[str, Dict[str, float]] = {}
    for bundle in data.get("data", {}).get("forecasts", []):
        model_id = bundle.get("model_id")
        hourly = bundle.get("hourly", [])
        if model_id:
            model_forecasts[model_id] = {
                h["time"]: h["temperature_2m_f"]
                for h in hourly
                if h.get("temperature_2m_f") is not None and h.get("time")
            }

    return model_forecasts


async def fetch_latest_observation(station_id: str) -> Optional[float]:
    """Fetch latest METAR observation running daily high.

    Returns daily_high_f or None.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{BASE_URL}/stations/{station_id}/observations/latest",
            headers=_API_HEADERS(),
        )
        resp.raise_for_status()
        data = resp.json()

    return data.get("data", {}).get("daily_high_f")


async def fetch_minutetemp_forecast(
    city_key: str,
    target_date: date,
    is_today: bool = True,
) -> Optional[dict]:
    """Main entry point. Fetches oracle scores, filters by bias, averages
    qualifying model forecasts, and returns an ensemble result dict.

    is_today=True  -> use day_of oracle scores
    is_today=False -> use day_ahead oracle scores
    """
    if not settings.MINUTETEMP_API_KEY:
        return None

    city_config = CITY_STATION_MAP.get(city_key)
    if not city_config:
        return None

    station_id = city_config["station_id"]
    mode = "day_of" if is_today else "day_ahead"

    try:
        qualified_models = await fetch_oracle_scores(station_id, mode)
        if not qualified_models:
            logger.warning(f"No qualifying models for {city_key} ({mode})")
            return None

        qualified_model_ids = {s["model_id"] for s in qualified_models}
        model_forecasts = await fetch_station_forecast(station_id, target_date)

        from datetime import timedelta
        date_start = target_date.isoformat()
        date_end = (target_date + timedelta(days=1)).isoformat()

        member_highs: List[float] = []
        models_used: List[str] = []
        for model_id in qualified_model_ids:
            temps_dict = model_forecasts.get(model_id, {})
            daily_temps = [
                v for k, v in temps_dict.items()
                if (k.startswith(date_start) or k.startswith(date_end))
                and v is not None
            ]
            if daily_temps:
                member_highs.append(max(daily_temps))
                models_used.append(model_id)

        if not member_highs:
            logger.warning(f"No forecast data for qualifying models in {city_key}")
            return None

        mean_high = statistics.mean(member_highs)
        std_high = statistics.stdev(member_highs) if len(member_highs) > 1 else 0.0

        metar_high: Optional[float] = None
        if is_today:
            try:
                metar_high = await fetch_latest_observation(station_id)
            except Exception as exc:
                logger.debug(f"METAR fetch failed for {station_id}: {exc}")

        logger.info(
            f"MinuteTemp {city_key} ({mode}): "
            f"mean={mean_high:.1f}F std={std_high:.1f}F "
            f"models={len(member_highs)} ({models_used}) "
            f"metar_high={metar_high}"
        )

        return {
            "mean_high":          mean_high,
            "std_high":           std_high,
            "num_members":        len(member_highs),
            "member_highs":       member_highs,
            "models_used":        models_used,
            "current_metar_high": metar_high,
            "mean_low":           None,
            "std_low":            None,
        }

    except Exception as exc:
        logger.warning(f"MinuteTemp fetch failed for {city_key}: {exc}")
        return None


async def fetch_current_observation(city_key: str) -> Optional[dict]:
    """Fetch the current METAR observation and running daily high/low for a city.

    Used by the GET /api/weather/observations endpoint.
    Returns a dict with current_temp_f, daily_high_f, daily_low_f, timestamp,
    station_id — or None on error.
    """
    city = CITY_STATION_MAP.get(city_key)
    if not city:
        return None

    station_id = city["station_id"]
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BASE_URL}/stations/{station_id}/observations/latest",
                headers=_API_HEADERS(),
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning(f"MinuteTemp observation fetch failed for {city_key}: {exc}")
        return None

    obs_data = data.get("data", {}) if data else {}
    observation = obs_data.get("observation", {})

    return {
        "current_temp_f": observation.get("temperature_f"),
        "daily_high_f":   obs_data.get("daily_high_f"),
        "daily_low_f":    obs_data.get("daily_low_f"),
        "timestamp":      observation.get("time"),
        "station_id":     station_id,
    }
