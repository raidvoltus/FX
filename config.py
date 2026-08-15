# config.py
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    """
    Konfigurasi Sentral untuk Autonomous Forex Trading Bot.
    Menggunakan frozen dataclass agar parameter tidak bisa diubah secara tak sengaja saat runtime.
    """

    # --- File Paths ---
    OANDA_CFG_PATH: str = os.getenv("OANDA_CFG_PATH", "oanda.cfg")
    ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", "models/ml_strategy_selector.pkl")
    STATE_STORE_PATH: str = os.getenv("STATE_STORE_PATH", "data/system_state.json")
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "autotrader.log")

    # --- System & Timing Settings ---
    LOOP_INTERVAL_SECONDS: int = int(os.getenv("LOOP_INTERVAL_SECONDS", "60"))  # Interval eksekusi main loop
    DEFAULT_GRANULARITY: str = os.getenv("DEFAULT_GRANULARITY", "M5")            # Granularitas default (e.g. M1, M5, H1)
    HISTORICAL_CANDLES_COUNT: int = int(os.getenv("HISTORICAL_CANDLES_COUNT", "500"))

    # --- Instrument Filters (Liquidity & Quality Control) ---
    ALLOWED_INSTRUMENTS: tuple = (
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", 
        "USD_CAD", "USD_CHF", "EUR_GBP", "EUR_JPY"
    )
    MAX_SPREAD_PIPS: float = float(os.getenv("MAX_SPREAD_PIPS", "3.0"))  # Tolak pair jika spread terlalu lebar

    # --- Machine Learning Engine Settings ---
    MIN_ML_CONFIDENCE: float = float(os.getenv("MIN_ML_CONFIDENCE", "0.65")) # Threshold minimum sinyal ML (65%)
    REQUIRED_LAG_FEATURE_COUNT: int = int(os.getenv("REQUIRED_LAG_FEATURE_COUNT", "6"))

    # --- Risk Engine Settings (DETERMINISTIC & STRICT) ---
    MAX_RISK_PER_TRADE_PCT: float = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "0.01")) # Maksimal 1% modal per trade
    MAX_PORTFOLIO_RISK_PCT: float = float(os.getenv("MAX_PORTFOLIO_RISK_PCT", "0.03")) # Maksimal 3% modal total ter-expose
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "3"))                # Maksimal posisi bersamaan
    
    # --- Volatility & Stop Loss Settings ---
    ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", "14"))
    ATR_STOP_LOSS_MULTIPLIER: float = float(os.getenv("ATR_STOP_LOSS_MULTIPLIER", "1.5")) # SL = 1.5 * ATR
    MIN_RISK_REWARD_RATIO: float = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.5"))      # Minimal R:R 1:1.5

    @classmethod
    def load(cls) -> "Config":
        """Factory method untuk memuat konfigurasi."""
        return cls()
