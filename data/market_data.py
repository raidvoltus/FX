"""
================================================================================
MODULE: data/market_data.py
DESCRIPTION: Production Synchronized Quantitative Data Ingestion Engine for OANDA Forex.
VERSION: 2026.1.0 (OANDA Forex Data Engine & Quality Control Edition)
PYTHON VERSION: 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Decoupled Ingestion, Resilient Data Provenance
================================================================================
"""

import os
import time
import logging
import hashlib
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

# Import internal wrapper broker (Step 3) & Config (Step 2)
from broker.oanda_client import OandaClient
from config import Config

logger = logging.getLogger("Forex.Data")


# =============================================================================
# 1. HIGH PRECISION TIMER & PROFILING UTILITIES
# =============================================================================
class HighPrecisionTimer:
    """
    High-precision timer untuk mengukur latensi ingestion data pasar.
    """
    def __init__(self, step_name: str = "ForexDataOperation"):
        self.step_name = step_name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self._execution_duration: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self._execution_duration = self.end_time - self.start_time
        if exc_type:
            logger.error(f"❌ [TIMER_FAIL] {self.step_name} gagal setelah {self._execution_duration:.4f}s")
        else:
            logger.info(f"⏱ [TIMER_SUCCESS] {self.step_name} selesai dalam {self._execution_duration:.4f}s")

    @property
    def execution_duration(self) -> float:
        if self.start_time > 0 and self.end_time == 0.0:
            return time.perf_counter() - self.start_time
        return self._execution_duration


# =============================================================================
# 2. LOCAL PARQUET DATA CACHE MANAGER
# =============================================================================
class ForexDataCacheManager:
    """
    Menangani caching lokal berbasis Parquet menggunakan Polars 
    untuk menghemat kuota API OANDA dan mempercepat proses backtest/inference.
    """
    def __init__(self, cache_dir: str = ".cache_forex_data", ttl_seconds: int = 300):
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, cache_key: str) -> Path:
        hashed_key = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"forex_klines_{hashed_key}.parquet"

    def read_cache(self, cache_key: str) -> Optional[pl.DataFrame]:
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None
        
        file_age = time.time() - cache_path.stat().st_mtime
        if file_age > self.ttl_seconds:
            logger.info(f"🔄 [CACHE_EXPIRED] Cache untuk {cache_key} sudah kadaluwarsa ({file_age:.0f}s).")
            return None
            
        try:
            df = pl.read_parquet(cache_path)
            logger.info(f"⚡ [CACHE_HIT] Berhasil memuat {df.height} baris data dari cache: {cache_path.name}")
            return df
        except Exception as e:
            logger.warning(f"⚠️ [CACHE_READ_ERR] Cache rusak {cache_path.name}: {e}")
            return None

    def write_cache(self, cache_key: str, df: pl.DataFrame) -> None:
        if df.height == 0:
            return
        cache_path = self._get_cache_path(cache_key)
        try:
            df.write_parquet(cache_path)
            logger.info(f"💾 [CACHE_STORE] Menyimpan {df.height} baris data ke {cache_path.name}")
        except Exception as e:
            logger.error(f"❌ [CACHE_WRITE_ERR] Gagal menyimpan cache: {e}")


# =============================================================================
# 3. UNIFIED FOREX MARKET DATA FETCHER & FILTER
# =============================================================================
class MarketDataFetcher:
    """
    Engine Data Utama untuk OANDA Forex.
    Menangani penyaringan instrumen (Spread & Likuiditas) dan Normalisasi Data.
    """

    def __init__(self, broker_client: OandaClient, enable_cache: bool = True):
        self.broker = broker_client
        self.config = Config.load()
        self.cache_mgr = ForexDataCacheManager(ttl_seconds=300)
        self.enable_cache = enable_cache

    def filter_tradeable_instruments(self, available_instruments: List[tuple]) -> List[str]:
        """
        Menyaring instrumen OANDA berdasarkan:
        1. Config ALLOWED_INSTRUMENTS (Pasangan mata uang utama)
        2. Filter Spread Maksimum (Mencegah trading saat spread meledak)
        """
        valid_pairs = []
        logger.info("🔍 Menjalankan Pemindaian & Penyaringan Spread Pasar...")

        for inst in available_instruments:
            # inst format dari tpqoa: ('EUR/USD', 'EUR_USD')
            symbol = inst[1] if isinstance(inst, (tuple, list)) and len(inst) > 1 else str(inst)

            if symbol not in self.config.ALLOWED_INSTRUMENTS:
                continue

            try:
                quote = self.broker.get_current_ask_bid(symbol)
                spread_pips = quote["spread"] * (10000 if "JPY" not in symbol else 100)

                if spread_pips <= self.config.MAX_SPREAD_PIPS:
                    valid_pairs.append(symbol)
                    logger.info(f"  ✔ [{symbol}] Spread: {spread_pips:.2f} pips (Lolos Filter)")
                else:
                    logger.warning(f"  ⚠️ [{symbol}] Spread terlalu lebar: {spread_pips:.2f} pips > Limit ({self.config.MAX_SPREAD_PIPS} pips)")
            except Exception as e:
                logger.error(f"  ❌ Gagal mengecek kuotasi {symbol}: {e}")

        logger.info(f"📈 Total Pasangan Forex Layak Diperdagangkan: {len(valid_pairs)} pairs")
        return valid_pairs

    def get_candles(
        self, 
        instrument: str, 
        granularity: str = "M5", 
        count: int = 500,
        use_cache: bool = True
    ) -> pl.DataFrame:
        """
        Mengambil data candle dari OANDA dan mengubahnya menjadi Polars DataFrame 
        dengan skema terstandarisasi.
        """
        cache_key = f"{instrument}_{granularity}_{count}"

        if self.enable_cache and use_cache:
            cached_df = self.cache_mgr.read_cache(cache_key)
            if cached_df is not None:
                return cached_df

        timer = HighPrecisionTimer(f"FetchCandles_{instrument}")
        with timer:
            try:
                # Mengambil histori menggunakan OandaClient
                # OANDA get_history menghasilkan pandas DataFrame
                end_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                # Menggunakan hitungan mundur atau rentang default
                df_pd = self.broker.api.get_history(
                    instrument=instrument,
                    granularity=granularity,
                    count=count,
                    price="M"  # Mid price
                )

                if df_pd is None or df_pd.empty:
                    logger.error(f"🛑 [DATA_FETCH_EMPTY] OANDA mengembalikan 0 record untuk {instrument}")
                    return pl.DataFrame()

                # Convert Pandas to Polars
                df_pd = df_pd.reset_index()
                df = pl.from_pandas(df_pd)

                # Rename & Normalisasi Kolom OANDA
                col_rename_map = {}
                for col in df.columns:
                    c_lower = col.lower()
                    if c_lower in ["time", "date", "timestamp"]:
                        col_rename_map[col] = "timestamp"
                    elif c_lower in ["c", "close", "midc"]:
                        col_rename_map[col] = "close"
                    elif c_lower in ["o", "open", "mido"]:
                        col_rename_map[col] = "open"
                    elif c_lower in ["h", "high", "midh"]:
                        col_rename_map[col] = "high"
                    elif c_lower in ["l", "low", "midl"]:
                        col_rename_map[col] = "low"
                    elif c_lower in ["v", "volume"]:
                        col_rename_map[col] = "volume"

                df = df.rename(col_rename_map)

                # Standard Guard Schema & Datetime Casting
                df = df.with_columns([
                    pl.lit(instrument).alias("ticker"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                    ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("returns_clean")
                ])

                # Pastikan timestamp ter-parse sebagai Datetime
                if "timestamp" in df.columns:
                    df = df.with_columns(
                        pl.col("timestamp").cast(pl.Datetime).alias("datetime"),
                        pl.col("timestamp").cast(pl.Datetime).dt.date().alias("date")
                    )

                if self.enable_cache and df.height > 0:
                    self.cache_mgr.write_cache(cache_key, df)

                logger.info(f"✔ [DATA_ENGINE_SUCCESS] {instrument} - {df.height} candle siap dalam {timer.execution_duration:.2f}s")
                return df

            except Exception as e:
                logger.error(f"❌ [DATA_ENGINE_ERROR] Gagal memuat data candle {instrument}: {e}")
                return pl.DataFrame()
