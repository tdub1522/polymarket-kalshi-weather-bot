"""Weather data fetcher using Open-Meteo Ensemble API and NWS observations."""
import httpx
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional
import statistics
import time

logger = logging.getLogger("trading_bot")

# City configurations keyed to exact NWS station locations used by Kalshi for settlement
CITY_CONFIG: Dict[str, dict] = {
    "los_angeles": {
        "lat": 33.9381, "lon": -118.3889,
        "name": "Los Angeles",
        "station": "KLAX",
        "location": "LAX International Airport",
    },
    "chicago": {
        "lat": 41.7842, "lon": -87.7553,
        "name": "Chicago",
        "station": "KMDW",
        "location": "Chicago Midway Airport",
    },
    "denver": {
        "lat": 39.8466, "lon": -104.6562,
        "name": "Denver",
        "station": "KDEN",
        "location": "Denver International Airport",
    },
    "dallas": {
        "lat": 32.8974, "lon": -97.0220,
        "name": "Dallas",
        "station": "KDFW",
        "location": "Dallas/Fort Worth International Airport",
    },
    "houston": {
        "lat": 29.6375, "lon": -95.2825,
        "name": "Houston",
        "station": "KHOU",
        "location": "Houston Hobby Airport",
    },
    "las_vegas": {
        "lat": 36.0719, "lon": -115.1634,
        "name": "Las Vegas",
        "station": "KLAS",
        "location": "Harry Reid International Airport",
    },
    "minneapolis": {
        "lat": 44.8831, "lon": -93.2289,
        "name": "Minneapolis",
        "station": "KMSP",
        "location": "Minneapolis-St. Paul International Airport",
    },
    "new_orleans": {
        "lat": 29.9928, "lon": -90.2508,
        "name": "New Orleans",
        "station": "KMSY",
        "location": "New Orleans International Airport",
    },
    "philadelphia": {
        "lat": 39.8733, "lon": -75.2268,
        "name": "Philadelphia",
        "station": "KPHL",
        "location": "Philadelphia International Airport",
    },
    "san_antonio": {
        "lat": 29.5328, "lon": -98.4636,
        "name": "San Antonio",
        "station": "KSAT",
        "location": "San Antonio International Airport",
    },
    "washington_dc": {
        "lat": 38.8483, "lon": -77.0342,
        "name": "Washington DC",
        "station": "KDCA",
        "location": "Reagan National Airport",
    },
}


@dataclass
class EnsembleForecast:
    """Ensemble weather forecast with per-member data."""
    city_key: str
    city_name: str
    target_date: date
    member_highs: List[float]  # Daily max temps (F) per ensemble member
    member_lows: List[float]   # Daily min temps (F) per ensemble member
    mean_high: float = 0.0
    std_high: float = 0.0
    mean_low: float = 0.0
    std_low: float = 0.0
    num_members: int = 0
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    models_used: List[str] = field(default_factory=list)
    current_metar_high: Optional[float] = None

    def __post_init__(self):
        if self.member_highs:
            self.num_members = len(self.member_highs)

    def probability_high_above(self, threshold_f: float) -> float:
        """Fraction of ensemble members with daily high above threshold."""
        if not self.member_highs:
            return 0.5
        count = sum(1 for h in self.member_highs if h > threshold_f)
        return count / len(self.member_highs)

    def probability_high_below(self, threshold_f: float) -> float:
        """Fraction of ensemble members with daily high below threshold."""
        return 1.0 - self.probability_high_above(threshold_f)

    def probability_low_above(self, threshold_f: float) -> float:
        """Fraction of ensemble members with daily low above threshold."""
        if not self.member_lows:
            return 0.5
        count = sum(1 for l in self.member_lows if l > threshold_f)
        return count / len(self.member_lows)

    def probability_low_below(self, threshold_f: float) -> float:
        """Fraction of ensemble members with daily low below threshold."""
        return 1.0 - self.probability_low_above(threshold_f)

    @property
    def ensemble_agreement(self) -> float:
        """How one-sided the ensemble is (0.5 = split, 1.0 = unanimous)."""
        if not self.member_highs:
            return 0.5
        median = statistics.median(self.member_highs)
        above = sum(1 for h in self.member_highs if h > median)
        frac = above / len(self.member_highs)
        return max(frac, 1 - frac)


# Simple cache: (city_key, target_date_str) -> (timestamp, EnsembleForecast)
_forecast_cache: Dict[str, tuple] = {}
_CACHE_TTL = 900  # 15 minutes


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _log_forecast_history(city_key: str, target_date: date,
                           forecast: EnsembleForecast, source: str) -> None:
    import json
    from backend.models.database import SessionLocal, ForecastHistory
    db = SessionLocal()
    try:
        record = ForecastHistory(
            city_key=city_key,
            station_id=CITY_CONFIG.get(city_key, {}).get("station", ""),
            target_date=target_date.isoformat(),
            mean_high=forecast.mean_high,
            std_high=forecast.std_high,
            num_members=forecast.num_members,
            models_used=json.dumps(getattr(forecast, "models_used", [])),
            member_highs=json.dumps([round(h, 2) for h in forecast.member_highs]),
            forecast_source=source,
        )
        db.add(record)
        db.commit()
    except Exception as e:
        logger.debug(f"Failed to log forecast history: {e}")
        db.rollback()
    finally:
        db.close()


async def fetch_ensemble_forecast(city_key: str, target_date: Optional[date] = None) -> Optional[EnsembleForecast]:
    """Fetch MinuteTemp oracle ensemble forecast. Returns None if unavailable."""
    if city_key not in CITY_CONFIG:
        logger.warning(f"Unknown city key: {city_key}")
        return None

    if target_date is None:
        target_date = date.today()

    cache_key = f"{city_key}_{target_date.isoformat()}"
    now = time.time()
    if cache_key in _forecast_cache:
        cached_time, cached_forecast = _forecast_cache[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_forecast

    from backend.config import settings as _settings
    from backend.data.minutetemp_client import fetch_minutetemp_forecast

    try:
        mt = await fetch_minutetemp_forecast(city_key, target_date)
        if not mt:
            logger.warning(f"MinuteTemp returned no data for {city_key} on {target_date}")
            return None

        logger.info(f"[MinuteTemp] {city_key} {target_date}: mean={mt['mean_high']:.1f}F models={mt.get('num_members', 0)}")
        config = CITY_CONFIG[city_key]
        forecast = EnsembleForecast(
            city_key=city_key,
            city_name=config["name"],
            target_date=target_date,
            member_highs=mt["member_highs"],
            member_lows=[],
            mean_high=mt["mean_high"],
            std_high=mt["std_high"],
            mean_low=0.0,
            std_low=0.0,
            models_used=mt.get("models_used", []),
            current_metar_high=mt.get("current_metar_high"),
        )
        _forecast_cache[cache_key] = (now, forecast)
        _log_forecast_history(city_key, target_date, forecast, "minutetemp")
        return forecast

    except Exception as exc:
        logger.warning(f"MinuteTemp forecast failed for {city_key}: {exc}")
        return None


async def fetch_nws_observed_temperature(city_key: str, target_date: Optional[date] = None) -> Optional[Dict[str, float]]:
    """
    Fetch observed temperature from NWS API for settlement.
    Returns dict with 'high' and 'low' in Fahrenheit, or None if not available.
    """
    if city_key not in CITY_CONFIG:
        return None

    city = CITY_CONFIG[city_key]

    if target_date is None:
        target_date = date.today()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # NWS observations endpoint
            station = city["station"]
            url = f"https://api.weather.gov/stations/{station}/observations"
            headers = {"User-Agent": "(trading-bot, contact@example.com)"}

            # Get observations for the target date
            start = datetime.combine(target_date, datetime.min.time()).isoformat() + "Z"
            end = datetime.combine(target_date + timedelta(days=1), datetime.min.time()).isoformat() + "Z"

            response = await client.get(url, params={"start": start, "end": end}, headers=headers)
            response.raise_for_status()
            data = response.json()

            features = data.get("features", [])
            if not features:
                return None

            temps = []
            for obs in features:
                props = obs.get("properties", {})
                temp_c = props.get("temperature", {}).get("value")
                if temp_c is not None:
                    temps.append(_celsius_to_fahrenheit(temp_c))

            if not temps:
                return None

            return {
                "high": max(temps),
                "low": min(temps),
            }

    except Exception as e:
        logger.warning(f"Failed to fetch NWS observations for {city_key}: {e}")
        return None
