"""Configuration settings for the BTC 5-min trading bot."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database (SQLite for Phase 1, PostgreSQL for production)
    DATABASE_URL: str = "sqlite:////tmp/tradingbot.db"

    # Kalshi API
    KALSHI_API_KEY_ID: Optional[str] = "86d34ae3-04f4-4b7b-92ae-663fbb636aa8"
    KALSHI_PRIVATE_KEY_PATH: Optional[str] = "/Users/treywoolley/.kalshi/kalshi-prod.pem"
    KALSHI_ENABLED: bool = True

    # BTC scanning
    SCAN_INTERVAL_SECONDS: int = 60
    SETTLEMENT_INTERVAL_SECONDS: int = 1800

    # Weather trading settings
    WEATHER_ENABLED: bool = True
    WEATHER_SCAN_INTERVAL_SECONDS: int = 300  # 5 min
    WEATHER_SETTLEMENT_INTERVAL_SECONDS: int = 1800  # 30 min
    WEATHER_MIN_EDGE_THRESHOLD: float = 0.08  # 8% — weather has more signal than 5-min BTC
    WEATHER_MAX_ENTRY_PRICE: float = 0.97
    WEATHER_MAX_TRADE_SIZE: float = 100.0
    WEATHER_CITIES: str = "nyc,chicago,miami,los_angeles,denver,philadelphia"

    # Sizing
    INITIAL_BANKROLL: float = 80.0
    KELLY_FRACTION: float = 0.25
    MIN_EDGE_THRESHOLD: float = 0.08

    # Safety guards
    SIMULATION_MODE: bool = True
    TRADING_ENABLED: bool = False
    DAILY_LOSS_LIMIT: float = 20.0
    MAX_TRADE_SIZE: float = 5.0
    MAX_OPEN_POSITIONS: int = 5

    # Discord alerts
    DISCORD_WEBHOOK_URL: Optional[str] = None
    DISCORD_ENABLED: bool = False

    class Config:
        env_file = ".env"


settings = Settings()

