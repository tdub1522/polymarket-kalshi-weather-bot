"""MinuteTemp API client — oracle-ranked ASOS weather forecasts and observations.

The oracle-score endpoint ranks models by recent accuracy; only models with
abs(high_bias) < 1.0 F are used when building the ensemble average, so the
signal reflects the most accurate models available for each station.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
from datetime import date
from typing import Dict, List, Optional

import httpx

from backend.config import settings

logger = logging.getLogger("trading_bot")

BASE_URL = "https://api.minutetemp.com/api/v1"

MODEL_ID_ALIASES = {
    "ecmwf_ifs025": "ecmwf_ifs",
    "ncep_gefs025": "ncep_gefs",
    "gfs_global_025": "gfs_global",
}

CITY_STATION_MAP: Dict[str, dict] = {
    "nyc":          {"station_id": "KNYC", "slug": "nyc"},
    "chicago":      {"station_id": "KMDW", "slug": "chi"},
    "miami":        {"station_id": "KMIA", "slug": "mia"},
    "los_angeles":  {"station_id": "KLAX", "slug": "lax"},
    "denver":       {"station_id": "KDEN", "slug": "den"},
    "philadelphia":  {"station_id": "KPHL", "slug": "phl"},
    "san_francisco": {"station_id": "KSFO", "slug": "sfo"},
    "minneapolis":   {"station_id": "KMSP", "slug": "msp"},
    "phoenix":       {"station_id": "KPHX", "slug": "phx"},
    "houston":       {"station_id": "KHOU", "slug": "hou"},
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
    qualified = sorted(scores, key=lambda s: s.get("high_mae", 999))[:5]

    model_summary = ", ".join(
        f"{s['model_name']} ({s['high_mae']:.2f})" for s in qualified
    )
    logger.info(
        f"Oracle scores {station_id} (day_of 7d): "
        f"top 5 by MAE selected: {model_summary}"
    )
    return qualified


async def fetch_station_forecast(station_id: str) -> Dict[str, List[float]]:
    """Fetch all model forecasts for a station.

    Returns dict keyed by model_id -> {time_str: temp_f} for all hours returned.
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
) -> Optional[dict]:
    """Fetch top-5 oracle models by MAE and return an ensemble forecast dict."""
    if not settings.MINUTETEMP_API_KEY:
        return None

    city_config = CITY_STATION_MAP.get(city_key)
    if not city_config:
        return None

    station_id = city_config["station_id"]
    mode = "day_of"

    try:
        qualified_models = await fetch_oracle_scores(station_id, mode)
        await asyncio.sleep(0.5)
        if not qualified_models:
            logger.warning(f"No qualifying models for {city_key} ({mode})")
            return None

        qualified_model_ids = {
            MODEL_ID_ALIASES.get(s["model_id"], s["model_id"])
            for s in qualified_models
        }
        model_forecasts = await fetch_station_forecast(station_id)

        member_highs: List[float] = []
        models_used: List[str] = []
        from datetime import timedelta
        date_start_utc = f"{target_date.isoformat()}T00:00:00Z"
        date_end_utc   = f"{(target_date + timedelta(days=1)).isoformat()}T12:00:00Z"

        for model_id in qualified_model_ids:
            temps_dict = model_forecasts.get(model_id, {})

            from datetime import datetime, timezone as _tz
            day_start = datetime(target_date.year, target_date.month, target_date.day,
                                 0, 0, 0, tzinfo=_tz.utc)
            day_end = day_start + timedelta(days=1, hours=12)

            daily_temps = []
            for k, v in temps_dict.items():
                if v is None:
                    continue
                try:
                    k_clean = k.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(k_clean)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_tz.utc)
                    dt_utc = dt.astimezone(_tz.utc)
                    if day_start <= dt_utc <= day_end:
                        daily_temps.append(v)
                except Exception:
                    continue
            if daily_temps:
                logger.info(
                    f"  {model_id}: {len(temps_dict)} total hours, "
                    f"{len(daily_temps)} in window [{date_start_utc} to {date_end_utc}], "
                    f"max={max(daily_temps):.1f}F"
                )
                member_highs.append(max(daily_temps))
                models_used.append(model_id)
            else:
                logger.info(
                    f"  {model_id}: {len(temps_dict)} total hours, 0 in window"
                )

        if not member_highs:
            logger.warning(f"No forecast data for qualifying models in {city_key}")
            return None

        mean_high = statistics.mean(member_highs)
        std_high = statistics.stdev(member_highs) if len(member_highs) > 1 else 0.0

        logger.info(
            f"MinuteTemp {city_key} (day_of top5): "
            f"mean={mean_high:.1f}F std={std_high:.1f}F "
            f"from {len(member_highs)} models: "
            f"{dict(zip(models_used, [round(h, 1) for h in member_highs]))}"
        )

        metar_high: Optional[float] = None
        try:
            metar_high = await fetch_latest_observation(station_id)
        except Exception as exc:
            logger.debug(f"METAR fetch failed for {station_id}: {exc}")

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
