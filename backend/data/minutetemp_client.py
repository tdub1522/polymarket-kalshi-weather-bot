"""MinuteTemp API client — settlement-grade ASOS weather forecasts and observations.

Replaces Open-Meteo as the forecast source when MINUTETEMP_ENABLED=True.
Returns EnsembleForecast objects compatible with the existing weather pipeline.
"""
from __future__ import annotations

import logging
import statistics
from datetime import date
from typing import Optional

import httpx

from backend.config import settings
from backend.data.weather import EnsembleForecast

logger = logging.getLogger("trading_bot")

BASE_URL = "https://api.minutetemp.com/api/v1"

CITY_STATION_MAP = {
    "nyc":          {"station_id": "KNYC", "slug": "nyc", "name": "New York"},
    "chicago":      {"station_id": "KMDW", "slug": "chi", "name": "Chicago"},
    "miami":        {"station_id": "KMIA", "slug": "mia", "name": "Miami"},
    "los_angeles":  {"station_id": "KLAX", "slug": "lax", "name": "Los Angeles"},
    "denver":       {"station_id": "KDEN", "slug": "den", "name": "Denver"},
    "philadelphia": {"station_id": "KPHL", "slug": "phl", "name": "Philadelphia"},
}


def _headers() -> dict:
    return {"X-API-Key": settings.MINUTETEMP_API_KEY or ""}


async def fetch_forecast(city_key: str, target_date: date) -> Optional[EnsembleForecast]:
    """Fetch multi-model temperature forecast from MinuteTemp for a city and date.

    Calls GET /api/v1/stations/{station_id}/forecast, extracts per-model daily
    highs from hourly temperature_2m_f data, applies 2-std outlier filtering,
    and returns an EnsembleForecast compatible with the existing weather pipeline.
    """
    city = CITY_STATION_MAP.get(city_key)
    if not city:
        logger.warning(f"MinuteTemp: unknown city key '{city_key}'")
        return None

    station_id = city["station_id"]
    date_str = str(target_date)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BASE_URL}/stations/{station_id}/forecast",
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning(f"MinuteTemp forecast fetch failed for {city_key}: {exc}")
        return None

    forecasts = data.get("data", {}).get("forecasts", [])
    member_highs: list[float] = []

    for bundle in forecasts:
        hourly = bundle.get("hourly", [])
        daily_temps: list[float] = []
        for h in hourly:
            if date_str in str(h.get("time", "")):
                temp = h.get("temperature_2m_f")
                if temp is not None:
                    daily_temps.append(float(temp))
        if daily_temps:
            member_highs.append(max(daily_temps))

    if not member_highs:
        logger.warning(f"MinuteTemp: no member highs found for {city_key} on {target_date}")
        return None

    mean = statistics.mean(member_highs)
    std = statistics.stdev(member_highs) if len(member_highs) > 1 else 0.0
    filtered = [m for m in member_highs if abs(m - mean) <= 2 * std] or member_highs

    final_mean = statistics.mean(filtered)
    final_std = statistics.stdev(filtered) if len(filtered) > 1 else 0.0

    logger.info(
        f"MinuteTemp forecast for {city_key} on {target_date}: "
        f"{final_mean:.1f}F +/- {final_std:.1f}F ({len(filtered)} models)"
    )

    return EnsembleForecast(
        city_key=city_key,
        city_name=city["name"],
        target_date=target_date,
        member_highs=filtered,
        member_lows=[],
        mean_high=final_mean,
        std_high=final_std,
        mean_low=0.0,
        std_low=0.0,
    )


async def fetch_current_observation(city_key: str) -> Optional[dict]:
    """Fetch the current METAR observation and running daily high/low for a city.

    Calls GET /api/v1/stations/{station_id}/observations/latest.
    Returns a dict with current_temp_f, daily_high_f, daily_low_f, timestamp,
    station_id — or None on error.
    """
    city = CITY_STATION_MAP.get(city_key)
    if not city:
        logger.warning(f"MinuteTemp: unknown city key '{city_key}'")
        return None

    station_id = city["station_id"]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BASE_URL}/stations/{station_id}/observations/latest",
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning(f"MinuteTemp observation fetch failed for {city_key}: {exc}")
        return None

    obs_data = data.get("data", {})
    observation = obs_data.get("observation", {})

    return {
        "current_temp_f": observation.get("temperature_f"),
        "daily_high_f":   obs_data.get("daily_high_f"),
        "daily_low_f":    obs_data.get("daily_low_f"),
        "timestamp":      observation.get("time"),
        "station_id":     station_id,
    }
