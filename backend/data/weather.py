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
    "nyc": {
        "lat": 40.7833,
        "lon": -73.9667,
        "name": "New York City",
        "station": "KNYC",
        "location": "Central Park",
    },
    "chicago": {
        "lat": 41.9797,
        "lon": -87.9044,
        "name": "Chicago",
        "station": "KORD",
        "location": "O'Hare International Airport",
    },
    "miami": {
        "lat": 25.7906,
        "lon": -80.3164,
        "name": "Miami",
        "station": "KMIA",
        "location": "Miami International Airport",
    },
    "los_angeles": {
        "lat": 33.9381,
        "lon": -118.3889,
        "name": "Los Angeles",
        "station": "KLAX",
        "location": "LAX International Airport",
    },
    "denver": {
        "lat": 39.8466,
        "lon": -104.6562,
        "name": "Denver",
        "station": "KDEN",
        "location": "Denver International Airport",
    },
    "boston": {
        "lat": 42.3606,
        "lon": -71.0100,
        "name": "Boston",
        "station": "KBOS",
        "location": "Boston Logan International Airport",
    },
    "philadelphia": {
        "lat": 39.8733,
        "lon": -75.2268,
        "name": "Philadelphia",
        "station": "KPHL",
        "location": "Philadelphia International Airport",
    },
    "atlanta": {
        "lat": 33.6407,
        "lon": -84.4277,
        "name": "Atlanta",
        "station": "KATL",
        "location": "Hartsfield-Jackson Atlanta International Airport",
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


ENSEMBLE_MODELS = [
    {"name": "icon_seamless",   "label": "DWD ICON EPS"},
    {"name": "ncep_gefs025",    "label": "GFS Ensemble 0.25"},
    {"name": "ecmwf_aifs025",   "label": "ECMWF AIFS 0.25"},
]


async def fetch_ensemble_forecast(city_key: str, target_date: Optional[date] = None) -> Optional[EnsembleForecast]:
    """
    Fetch ensemble forecast from Open-Meteo combining ICON EPS, GFS, and ECMWF AIFS.
    Returns per-member daily max temperatures in Fahrenheit.
    """
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

    config = CITY_CONFIG[city_key]
    logger.info(f"Fetching multi-model ensemble for {city_key} using {config['station']} ({config['location']}) at lat={config['lat']}, lon={config['lon']}")

    member_highs: List[float] = []
    member_lows: List[float] = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        for model in ENSEMBLE_MODELS:
            params = {
                "latitude": config["lat"],
                "longitude": config["lon"],
                "hourly": "temperature_2m",
                "temperature_unit": "fahrenheit",
                "timezone": "America/New_York",
                "start_date": target_date.isoformat(),
                "end_date": target_date.isoformat(),
                "models": model["name"],
            }
            try:
                response = await client.get(
                    "https://ensemble-api.open-meteo.com/v1/ensemble",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                hourly = data.get("hourly", {})
                highs_from_model: List[float] = []
                lows_from_model: List[float] = []

                for key, values in hourly.items():
                    if "temperature_2m" in key and values:
                        valid = [v for v in values if v is not None]
                        if valid:
                            highs_from_model.append(max(valid))
                            lows_from_model.append(min(valid))

                logger.info(f"{model['label']} response keys: {list(hourly.keys())[:5]}")
                logger.info(f"{model['label']} members found: {len(highs_from_model)}")
                member_highs.extend(highs_from_model)
                member_lows.extend(lows_from_model)

            except Exception as e:
                logger.warning(f"Failed to fetch {model['label']} for {city_key}: {e}")

    logger.info(f"Total members combined: {len(member_highs)}")
    logger.info(f"Sample member highs: {member_highs[:5]}")

    if not member_highs:
        logger.warning(f"No ensemble members found for {city_key} on {target_date}")
        return None

    initial_mean_high = statistics.mean(member_highs)
    initial_std_high = statistics.stdev(member_highs) if len(member_highs) > 1 else 0.0
    filtered_highs = [m for m in member_highs if abs(m - initial_mean_high) <= 2 * initial_std_high]
    mean_high = statistics.mean(filtered_highs)
    std_high = statistics.stdev(filtered_highs) if len(filtered_highs) > 1 else 0.0
    logger.info(f"High temp filtering: {len(filtered_highs)}/{len(member_highs)} members within 2 std")

    initial_mean_low = statistics.mean(member_lows) if member_lows else 0.0
    initial_std_low = statistics.stdev(member_lows) if len(member_lows) > 1 else 0.0
    filtered_lows = [m for m in member_lows if abs(m - initial_mean_low) <= 2 * initial_std_low] if member_lows else []
    mean_low = statistics.mean(filtered_lows) if filtered_lows else 0.0
    std_low = statistics.stdev(filtered_lows) if len(filtered_lows) > 1 else 0.0
    logger.info(f"Low temp filtering: {len(filtered_lows)}/{len(member_lows)} members within 2 std")

    forecast = EnsembleForecast(
        city_key=city_key,
        city_name=config["name"],
        target_date=target_date,
        member_highs=filtered_highs,
        member_lows=filtered_lows,
        mean_high=mean_high,
        std_high=std_high,
        mean_low=mean_low,
        std_low=std_low,
    )

    _forecast_cache[cache_key] = (now, forecast)
    logger.info(f"Ensemble forecast for {config['name']} on {target_date}: "
                f"High {forecast.mean_high:.1f}F +/- {forecast.std_high:.1f}F "
                f"({forecast.num_members} members total)")

    return forecast


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
