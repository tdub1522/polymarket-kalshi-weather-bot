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
    WEATHER_MAX_TRADE_SIZE: float = 15.0
    WEATHER_CITIES: str = "nyc,chicago,miami,los_angeles,denver,philadelphia"

    # Sizing
    INITIAL_BANKROLL: float = 80.0
    KELLY_FRACTION: float = 0.25
    MIN_EDGE_THRESHOLD: float = 0.08

    # Safety guards
    SIMULATION_MODE: bool = True
    TRADING_ENABLED: bool = False
    DAILY_LOSS_LIMIT: float = 20.0
    MAX_TRADE_SIZE: float = 15.0
    MAX_OPEN_POSITIONS: int = 5

    # Discord alerts
    DISCORD_WEBHOOK_URL: Optional[str] = None
    DISCORD_ENABLED: bool = False

    # ── KXBTC15M (Kalshi BTC 15-min) signal pipeline ─────────────────
    # Signal-only by default. Auto-execution is gated by TRADING_ENABLED
    # AND KXBTC15M_AUTO_EXECUTE; both must be True before any order code
    # path ever runs (and that code path is intentionally not wired up
    # in v1 — flip these flags only after edge is established).
    KXBTC15M_ENABLED: bool = False
    KXBTC15M_AUTO_EXECUTE: bool = False
    KXBTC15M_SCAN_INTERVAL_SECONDS: int = 300  # 5 min, same cadence as repo
    KXBTC15M_MIN_EDGE: float = 0.03            # 3% minimum edge (repo default)
    KXBTC15M_MAX_TRADE_SIZE: float = 15.0      # $ per signal — match weather cap
    KXBTC15M_DAILY_LOSS_CAP: float = 20.0      # $ per day across all KXBTC15M trades
    KXBTC15M_MAX_DRAWDOWN_PCT: float = 0.15    # bot pauses if equity drops 15%
    KXBTC15M_MAX_TRADES_PER_DAY: int = 24
    KXBTC15M_SEND_PASS_ALERTS: bool = False    # Only alert on actionable signals

    # Anthropic — used by ROMA pipeline. Falls back to deterministic GBM
    # heuristic if missing (pipeline still runs, just less informed).
    ANTHROPIC_API_KEY: Optional[str] = None

    # Separate webhook for BTC signals so it doesn't blend with weather.
    DISCORD_BTC_WEBHOOK_URL: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()

