"""MinuteTemp API client — settlement-grade ASOS weather forecasts and observations.

fetch_minutetemp_forecast() is the primary entry point; weather.py calls it and
wraps the result into EnsembleForecast so the rest of the pipeline is unchanged.
"""
from __future__ import annotations

import logging
import statistics
from datetime import date
from typing import Dict, Optional

import httpx

from backend.config import settings

logger = logging.getLogger("trading_bot")

BASE_URL = "https://api.minutetemp.com/api/v1"

CITY_STATION_MAP: Dict[str, dict] = {
    "nyc":          {"station_id": "KNYC", "slug": "nyc", "name": "New York"},
    "chicago":      {"station_id": "KMDW", "slug": "chi", "name": "Chicago"},
    "miami":        {"station_id": "KMIA", "slug": "mia", "name": "Miami"},
    "los_angeles":  {"station_id": "KLAX", "slug": "lax", "name": "Los Angeles"},
    "denver":       {"station_id": "KDEN", "slug": "den", "name": "Denver"},
    "philadelphia": {"station_id": "KPHL", "slug": "phl", "name": "Philadelphia"},
}

# Module-level cache so Discord can pull the latest running high without an extra API call.
_metar_highs: Dict[str, Optional[float]] = {}


class MinuteTempClient:
    def __init__(self):
        self.api_key = settings.MINUTETEMP_API_KEY
        self.headers = {"X-API-Key": self.api_key}

    async def get_forecast(self, station_id: str) -> Optional[dict]:
        """Fetch forecast from all 20 models for a station.

        GET /api/v1/stations/{station_id}/forecast
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BASE_URL}/stations/{station_id}/forecast",
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def get_latest_observation(self, station_id: str) -> Optional[dict]:
        """Fetch latest METAR observation including running daily high.

        GET /api/v1/stations/{station_id}/observations/latest
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BASE_URL}/stations/{station_id}/observations/latest",
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def get_oracle_scores(self, station_id: str) -> Optional[dict]:
        """Fetch oracle model accuracy scores to find the best-performing model.

        GET /api/v1/stations/{station_id}/oracle-scores
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BASE_URL}/stations/{station_id}/oracle-scores",
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()


async def fetch_minutetemp_forecast(city_key: str, target_date: date) -> Optional[dict]:
    """Fetch forecast for a city using MinuteTemp API.

    Returns a dict with mean_high, std_high, num_members, member_highs,
    current_metar_high (running daily high from METAR), mean_low, std_low.
    Falls back to None if MinuteTemp is not configured or the fetch fails.
    """
    if not settings.MINUTETEMP_API_KEY:
        return None

    city_config = CITY_STATION_MAP.get(city_key)
    if not city_config:
        return None

    station_id = city_config["station_id"]
    client = MinuteTempClient()

    try:
        forecast_data = await client.get_forecast(station_id)

        member_highs: list[float] = []
        if forecast_data and "data" in forecast_data:
            bundles = forecast_data["data"].get("forecasts", [])
            for bundle in bundles:
                hourly = bundle.get("hourly", [])
                if hourly:
                    temps = [
                        h.get("temperature_2m_f")
                        for h in hourly
                        if h.get("temperature_2m_f") is not None
                    ]
                    if temps:
                        member_highs.append(max(temps))

        if not member_highs:
            return None

        mean_high = statistics.mean(member_highs)
        std_high = statistics.stdev(member_highs) if len(member_highs) > 1 else 0.0

        obs_data = await client.get_latest_observation(station_id)
        current_metar_high: Optional[float] = None
        if obs_data and "data" in obs_data:
            obs = obs_data["data"]
            current_metar_high = obs.get("daily_high_f")

        _metar_highs[city_key] = current_metar_high

        logger.info(
            f"MinuteTemp {city_key} ({station_id}): "
            f"mean={mean_high:.1f}F std={std_high:.1f}F "
            f"models={len(member_highs)} metar_high={current_metar_high}F"
        )

        return {
            "mean_high":          mean_high,
            "std_high":           std_high,
            "num_members":        len(member_highs),
            "member_highs":       member_highs,
            "current_metar_high": current_metar_high,
            "mean_low":           None,
            "std_low":            None,
        }

    except Exception as exc:
        logger.warning(f"MinuteTemp fetch failed for {city_key}: {exc}")
        return None


async def fetch_current_observation(city_key: str) -> Optional[dict]:
    """Fetch the current METAR observation and running daily high/low for a city.

    Returns a dict with current_temp_f, daily_high_f, daily_low_f, timestamp,
    station_id — or None on error.
    """
    city = CITY_STATION_MAP.get(city_key)
    if not city:
        logger.warning(f"MinuteTemp: unknown city key '{city_key}'")
        return None

    station_id = city["station_id"]
    client = MinuteTempClient()

    try:
        resp_data = await client.get_latest_observation(station_id)
    except Exception as exc:
        logger.warning(f"MinuteTemp observation fetch failed for {city_key}: {exc}")
        return None

    obs_data = resp_data.get("data", {}) if resp_data else {}
    observation = obs_data.get("observation", {})

    return {
        "current_temp_f": observation.get("temperature_f"),
        "daily_high_f":   obs_data.get("daily_high_f"),
        "daily_low_f":    obs_data.get("daily_low_f"),
        "timestamp":      observation.get("time"),
        "station_id":     station_id,
    }
